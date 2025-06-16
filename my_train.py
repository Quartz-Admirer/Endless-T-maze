import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os

from RATE.RATE_model import RATE
from datacollect import load_trajectories
from conf import config

class EndlessTMazeDataset(Dataset):
    def __init__(self, trajectories, context_length, sections):
        self.trajectories = trajectories
        self.sections = sections
        self.max_length = context_length * sections
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        traj_len = len(traj['actions'])
        
        states = traj['states'][:-1]
        rtgs = traj['rtgs'][:-1]
        timesteps = traj['timesteps'][:-1]
        
        input_actions = np.concatenate([[-10], traj['actions'][:-1]])

        target_actions = traj['actions']

        if traj_len < self.max_length:
            pad_len = self.max_length - traj_len
            
            states = np.concatenate([states, np.zeros((pad_len, 4))])
            input_actions = np.concatenate([input_actions, np.ones(pad_len) * -10])
            target_actions = np.concatenate([target_actions, np.ones(pad_len) * -10])
            rtgs = np.concatenate([rtgs, np.zeros(pad_len)])
            timesteps = np.concatenate([timesteps, np.ones(pad_len) * -10])
            masks = np.concatenate([np.ones(traj_len), np.zeros(pad_len)])
        else:
            states = states[:self.max_length]
            input_actions = input_actions[:self.max_length]
            target_actions = target_actions[:self.max_length]
            rtgs = rtgs[:self.max_length]
            timesteps = timesteps[:self.max_length]
            masks = np.ones(self.max_length)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(input_actions).unsqueeze(-1),
            torch.FloatTensor(rtgs).unsqueeze(-1),
            torch.LongTensor(target_actions).unsqueeze(-1),
            torch.LongTensor(timesteps),
            torch.FloatTensor(masks)
        )

def train():
    torch.manual_seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])

    train_corridor_length = config["env"]["train_corridor_length"]
    train_num_corridors = config["env"]["train_num_corridors"]
    max_seq_len = train_num_corridors * (train_corridor_length + 1)
    config["training"]["max_seq_len"] = max_seq_len
    
    sections = config["training"]["sections"]
    if max_seq_len % sections != 0:
        raise ValueError(f"max_seq_len ({max_seq_len}) must be divisible by sections ({sections})")
    context_length = max_seq_len // sections
    config["training"]["context_length"] = context_length
    
    device = torch.device(config["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    
    trajectories = load_trajectories("endless_tmaze_trajectories.pkl")
    dataset = EndlessTMazeDataset(
        trajectories, 
        config["training"]["context_length"],
        config["training"]["sections"]
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True
    )
    
    model = RATE(**config["model"]).to(device)
    torch.nn.init.xavier_uniform_(model.r_w_bias)
    torch.nn.init.xavier_uniform_(model.r_r_bias)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(config["training"]["beta_1"], config["training"]["beta_2"])
    )
    
    model.train()
    tokens_processed = 0
    
    for epoch in range(config["training"]["epochs"]):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch in pbar:
            states, input_actions, rtgs, target_actions, timesteps, masks = batch
            
            batch_size = states.size(0)
            blocks_context = config["training"]["context_length"]
            
            last_segment_with_data = -1
            if config["training"]["log_last_segment_loss_only"]:
                for i in range(config["training"]["sections"] - 1, -1, -1):
                    if masks[:, i * blocks_context:(i + 1) * blocks_context].any():
                        last_segment_with_data = i
                        break
            
            memory = None
            mem_tokens = model.mem_tokens.repeat(1, batch_size, 1) if model.mem_tokens is not None else None
            
            total_loss = 0
            
            for block_idx in range(config["training"]["sections"]):
                from_idx = block_idx * blocks_context
                to_idx = (block_idx + 1) * blocks_context
                
                x = states[:, from_idx:to_idx, :].to(device)
                y_in = input_actions[:, from_idx:to_idx, :].to(device)
                r = rtgs[:, from_idx:to_idx, :].to(device)
                y_target = target_actions[:, from_idx:to_idx, :].to(device)
                t = timesteps[:, from_idx:to_idx].to(device)
                m = masks[:, from_idx:to_idx].to(device)
                
                if mem_tokens is not None:
                    mem_tokens = mem_tokens.detach()
                
                if memory is not None:
                    res = model(x, y_in, r, y_target, t, *memory, mem_tokens=mem_tokens, masks=m)
                else:
                    res = model(x, y_in, r, y_target, t, mem_tokens=mem_tokens, masks=m)
                
                logits = res['logits']
                memory = res.get('new_mems', None)
                mem_tokens = res.get('mem_tokens', None)

                should_calculate_loss = (
                    not config["training"]["log_last_segment_loss_only"] or 
                    (last_segment_with_data != -1 and block_idx == last_segment_with_data)
                )
                
                if should_calculate_loss:
                    if config["training"]["log_last_segment_loss_only"]:
                        logits_last_tokens = []
                        targets_last_tokens = []
                        pad_val = config["model"]["padding_idx"]

                        for b in range(batch_size):
                            tgt_seq = y_target[b].squeeze(-1)
                            valid_pos = (tgt_seq != pad_val).nonzero(as_tuple=True)[0]
                            if valid_pos.numel() == 0:
                                continue
                            last_idx = valid_pos[-1].item()
                            logits_last_tokens.append(logits[b, last_idx])
                            targets_last_tokens.append(tgt_seq[last_idx])

                        if logits_last_tokens:  # list not empty
                            logits_stack = torch.stack(logits_last_tokens, dim=0)
                            targets_stack = torch.stack(targets_last_tokens, dim=0).long()
                            loss = F.cross_entropy(logits_stack, targets_stack, reduction='mean')
                            total_loss += loss
                    else:
                        current_mask = (y_target != config["model"]["padding_idx"]).squeeze(-1)
                        if current_mask.any():
                            loss = F.cross_entropy(
                                logits[current_mask],
                                y_target[current_mask].long().squeeze(-1),
                                reduction='mean'
                            )
                            total_loss += loss
            
            if total_loss > 0:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_norm_clip"])
                optimizer.step()
            
                epoch_loss += total_loss.item()
                num_batches += 1
                
                tokens_processed += m.sum().item()
                lr = config["training"]["learning_rate"]
                
                pbar.set_postfix({'loss': total_loss.item() / config["training"]["sections"], 'lr': lr})
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            os.makedirs(config["experiment"]["save_path"], exist_ok=True)
            torch.save(model.state_dict(), f"{config['experiment']['save_path']}/model_epoch_{epoch+1}.pth")
    
    os.makedirs(config["experiment"]["save_path"], exist_ok=True)
    torch.save(model.state_dict(), f"{config['experiment']['save_path']}/model_final.pth")
    print("Training completed!")

if __name__ == "__main__":
    train()
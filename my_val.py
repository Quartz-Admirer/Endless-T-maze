import torch
import numpy as np
from tqdm import tqdm
import argparse

from RATE.RATE_model import RATE
from endless_tmaze import EndlessTMaze
from conf import config

@torch.no_grad()
def sample_action(model, states, actions, rtgs, timesteps, mem_tokens, saved_context):
    context_length = config["training"]["context_length"]
    
    
    states_cond = states if states.size(1) <= context_length else states[:, -context_length:]
    actions_cond = actions if actions is None or actions.size(1) <= context_length else actions[:, -context_length:]
    rtgs_cond = rtgs if rtgs.size(1) <= context_length else rtgs[:, -context_length:]
    timesteps_cond = timesteps if timesteps.size(1) <= context_length else timesteps[:, -context_length:]
    
    if saved_context is not None:
        results = model(states_cond, actions_cond, rtgs_cond, None, timesteps_cond, *saved_context, mem_tokens=mem_tokens)
    else:
        results = model(states_cond, actions_cond, rtgs_cond, None, timesteps_cond, mem_tokens=mem_tokens)
    
    logits = results['logits'][:, -1, :]
    memory = results.get('new_mems', None)
    mem_tokens = results.get('mem_tokens', None)
    
    probs = torch.softmax(logits, dim=-1)
    action = torch.argmax(probs, dim=-1)
    
    return action, mem_tokens, memory


def evaluate(model_path, num_episodes=100, corridor_length=30, num_corridors=3):
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
    
    model = RATE(**config["model"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_success = 0
    total_rewards = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        env = EndlessTMaze(
            corridor_length=corridor_length,
            num_corridors=num_corridors,
            seed=config["experiment"]["seed"] + episode + 1000
        )
        
        state = env.reset()
        done = False
        
        all_states = [np.array([state[1], state[2], 0, np.random.randint(-1, 2)])]
        all_actions = [-10]
        
        target_return = torch.tensor(config["env"]["goal_reward"] * num_corridors, device=device, dtype=torch.float32).reshape(1, 1, 1)
        timesteps = torch.zeros((1, 1, 1), device=device, dtype=torch.long)
        
        mem_tokens = model.mem_tokens.repeat(1, 1, 1) if hasattr(model, 'mem_tokens') and model.mem_tokens is not None else None
        saved_context = None
        new_mem_tokens, new_context = None, None
        
        episode_reward = 0
        
        for t in range((corridor_length + 1) * num_corridors):
            flag = 1 if env.x == env.corridor_length else 0
            noise = np.random.randint(-1, 2)
            current_state_to_model = np.array([state[1], state[2], flag, noise])

            if t == 0:
                # Replace the initial dummy state (already added) with correct one
                all_states[0] = current_state_to_model
            else:
                all_states.append(current_state_to_model)

            if t > 0 and t % context_length == 0:
                mem_tokens = new_mem_tokens
                saved_context = new_context
            
            pad_val = config["model"]["padding_idx"]
            all_actions.append(pad_val)

            states_tensor = torch.from_numpy(np.array(all_states)).float().reshape(1, -1, 4).to(device)
            actions_np = np.array(all_actions, dtype=np.int64).reshape(1, -1, 1)

            if t == 0:
                actions_tensor = None
            else:
                actions_tensor = torch.from_numpy(actions_np[:, 1:, :]).long().to(device)
            
            action_pred, new_mem_tokens, new_context = sample_action(
                model, states_tensor, actions_tensor, target_return, 
                timesteps, mem_tokens, saved_context
            )
            
            action = action_pred.item()
            all_actions[-1] = action
            
            state, reward, done, info = env.step(action)
            episode_reward += reward

            pred_return = target_return[:, -1, :] - reward
            target_return = torch.cat([target_return, pred_return.reshape(1, 1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            if done:
                break
        
        if episode_reward >= config["env"]["goal_reward"] * num_corridors:
            total_success += 1
        total_rewards.append(episode_reward)

    success_rate = total_success / num_episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Max Possible Reward: {config['env']['goal_reward'] * num_corridors}")
    
    return success_rate, avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=f"{config['experiment']['save_path']}/model_final.pth", 
                        help='Path to the model weights file.')
    parser.add_argument('--corridor_length', type=int, default=config["env"]["val_corridor_length"],
                        help='Length of the corridor for validation.')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to run for evaluation.')

    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        corridor_length=args.corridor_length,
        num_corridors=config["env"]["val_num_corridors"]
    )
config = {
    # Environment settings
    "env": {
        "train_corridor_length": 11,
        "val_corridor_length": 11,
        "test_corridor_length": 20,
        "train_num_corridors": 5,
        "val_num_corridors": 5,
        "goal_reward": 1.0,
        "penalty": 0.0,
    },
    
    # Model architecture
    "model": {
        "env_name": "tmaze",
        "state_dim": 4,
        "act_dim": 4,
        "n_layer": 10,
        "n_head": 4,
        "n_head_ca": 2,
        "d_model": 64,
        "d_head": 32,
        "d_inner": 32,
        "dropout": 0.2,
        "dropatt": 0.0,
        "mem_len": 0,
        "ext_len": 0,
        "num_mem_tokens": 5,
        "mem_at_end": True,
        "mrv_act": "relu",
        "skip_dec_ffn": True,
        "padding_idx": -10,
        "max_ep_len": 1000,
    },
    
    # Training settings
    "training": {
        "batch_size": 64,
        "learning_rate": 1e-3,
        "lr_end_factor": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "weight_decay": 0.0,
        "warmup_steps": 1000,
        "final_tokens": 10000000,
        "grad_norm_clip": 1.0,
        "epochs": 20,
        "sections": 3,
        "use_cosine_decay": True,
        "log_last_segment_loss_only": False,
    },
    
    # Data collection
    "data": {
        "num_trajectories": 1000,
        "gamma": 1.0,
    },

    # Experiment settings
    "experiment": {
        "seed": 42,
        "device": "cuda",
        "save_path": "./checkpoints",
        "log_interval": 10,
    },
    
    "model_mode": "RATE",
    "arch_mode": "TrXL",
}
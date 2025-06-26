import pickle
import numpy as np
from tqdm import tqdm
from endless_tmaze import EndlessTMaze

def oracle_policy(env):
    """Uses the environment's internal logic to get the perfect action."""
    return env.get_optimal_action()

def collect_trajectories(num_trajectories, max_corridor_length, num_corridors, seed):
    trajectories = []
    for i in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        current_corridor_length = np.random.randint(3, max_corridor_length + 1)

        env = EndlessTMaze(
            corridor_length=current_corridor_length,
            num_corridors=num_corridors,
            seed=seed + i
        )

        traj_states = []
        traj_actions = []
        traj_rewards = []
        
        state = env.reset()
        done = False

        while not done:
            action = oracle_policy(env)

            flag = 1 if env.x == env.corridor_length else 0
            noise = np.random.randint(-1, 2)
            state_to_save = np.array([state[1], state[2], flag, noise])
            
            traj_states.append(state_to_save)
            traj_actions.append(action)

            state, reward, done, _ = env.step(action)
            traj_rewards.append(reward)

        flag = 1 if env.x == env.corridor_length else 0
        noise = np.random.randint(-1, 2)
        final_state_to_save = np.array([state[1], state[2], flag, noise])
        traj_states.append(final_state_to_save)
        
        traj_rewards = np.array(traj_rewards)
        rtgs = np.zeros_like(traj_rewards, dtype=np.float32)
        current_rtg = 0
        for t in reversed(range(len(traj_rewards))):
            current_rtg += traj_rewards[t]
            rtgs[t] = current_rtg
            
        rtgs = np.append(rtgs, 0)

        trajectories.append({
            'states': np.array(traj_states, dtype=np.float32),
            'actions': np.array(traj_actions, dtype=np.int64),
            'rtgs': rtgs,
            'timesteps': np.arange(len(traj_states), dtype=np.int64)
        })
        
    return trajectories

def save_trajectories(trajectories, filename):
    with open(filename, 'wb') as f:
        pickle.dump(trajectories, f)

def load_trajectories(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    from conf import config
    
    trajectories = collect_trajectories(
        num_trajectories=config["data"]["num_trajectories"],
        max_corridor_length=config["env"]["train_corridor_length"],
        num_corridors=config["env"]["train_num_corridors"],
        seed=config["experiment"]["seed"]
    )
    
    save_trajectories(trajectories, "endless_tmaze_trajectories.pkl")
    print(f"Collected {len(trajectories)} trajectories with variable corridor lengths.")
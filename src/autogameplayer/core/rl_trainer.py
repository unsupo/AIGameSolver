import os
import argparse
from pathlib import Path
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

from autogameplayer.core.gym_env import UniversalRLWrapper
from autogameplayer.core.config import settings

class TrainingCallback(BaseCallback):
    """Callback for saving models and logging."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.save_path = settings.base_dir / "models" / "rl_best_model.zip"

    def _on_step(self) -> bool:
        # Save every 5000 steps
        if self.num_timesteps % 5000 == 0:
            os.makedirs(self.save_path.parent, exist_ok=True)
            self.model.save(str(self.save_path))
            print(f"💾 Saved RL model at {self.num_timesteps} steps.")
        return True

def make_env(config_path: str, rank: int, seed: int = 0):
    """Utility function for multiprocess env."""
    def _init():
        os.environ["AGP_WORKER_ID"] = str(rank)
        env = UniversalRLWrapper(config_path=config_path)
        # We can seed it if needed
        return env
    return _init

def train_rl(config_path: str, num_envs: int = 4, total_timesteps: int = 100000):
    print(f"🚀 Starting Deep RL (RecurrentPPO) Training with {num_envs} workers...")
    
    # Create vectorized environment for massive parallelism
    env = SubprocVecEnv([make_env(config_path, i) for i in range(num_envs)])
    model = None
    
    try:
        # Initialize RecurrentPPO (PPO + LSTM)
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0003,
            n_steps=256, # Steps per worker before updating
            batch_size=64,
            tensorboard_log=str(settings.base_dir / "logs" / "tensorboard")
        )
        
        callback = TrainingCallback()
        
        print("🧠 Model initialized. Beginning training loop. (Press Ctrl+C to stop safely)")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
    except KeyboardInterrupt:
        print("\n🛑 RL Training Interrupted! Saving progress...")
    finally:
        os.makedirs(settings.base_dir / "models", exist_ok=True)
        if model:
            model.save(str(settings.base_dir / "models" / "rl_latest_model.zip"))
        env.close()
        print("✅ Graceful shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="Deep RL Trainer (PPO+LSTM)")
    parser.add_argument("--config", type=str, help="Path to game config YAML")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel emulators")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training steps")
    args = parser.parse_args()

    config_path = args.config
    if not config_path:
        # Better fallback: look for ANY yaml in configs/
        yaml_files = list(Path(settings.base_dir / "configs").glob("*.yaml"))
        if yaml_files:
            config_path = str(yaml_files[0])
        else:
            print("❌ Error: No configuration file found in ./configs/")
            return
        
    train_rl(config_path, num_envs=args.workers, total_timesteps=args.timesteps)

if __name__ == "__main__":
    main()

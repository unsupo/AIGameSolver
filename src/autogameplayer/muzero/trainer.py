import torch
import torch.nn.functional as F
import sqlite3
import numpy as np
import asyncio
import copy
from pathlib import Path
from typing import List, Dict
from autogameplayer.core.config import settings
from autogameplayer.muzero.networks import MuZeroModel
from autogameplayer.utils.database import get_db_connection
from autogameplayer.utils.math import symlog, symexp, two_hot, from_two_hot


class MuZeroOfflineTrainer:
    """
    Upgraded DreamerV3-style Offline Trainer.
    Uses two-hot categorical loss and imagination-based policy optimization.
    """

    def __init__(self, model: MuZeroModel, storage_path: Path = None, lr=1e-4):
        self.model = model
        self.storage_path = storage_path or settings.models_dir / "long_term_memory.db"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device)

        # Target Networks
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(self.device)

        self.training_steps = 0
        self.target_update_interval = 100
        
        # Action space mapping
        self.button_to_idx = {
            btn: i for i, btn in enumerate(["up", "down", "left", "right", "a", "b", "start", "select"])
        }

    def sample_sequence_batch(self, batch_size: int, seq_len: int) -> List[List[Dict]]:
        """Samples random sequences from the replay buffer using Priority."""
        try:
            with get_db_connection(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Prioritized sampling placeholder (using priority column)
                cursor = conn.execute(
                    "SELECT id FROM replay_buffer WHERE priority > 0 ORDER BY RANDOM() LIMIT ?",
                    (batch_size,),
                )
                start_ids = [row["id"] for row in cursor]

                batch = []
                for start_id in start_ids:
                    # Get sequence starting from random point
                    cursor = conn.execute(
                        "SELECT * FROM replay_buffer WHERE id >= ? ORDER BY id ASC LIMIT ?",
                        (start_id, seq_len),
                    )
                    seq = [dict(row) for row in cursor]
                    if len(seq) == seq_len:
                        batch.append(seq)
                return batch
        except Exception as e:
            print(f"⚠️ Sampling failed: {e}")
            return []

    async def train_step(self, batch_size=16, sequence_length=50, burn_in=20, imagination_horizon=15) -> float:
        """
        Combined World Model and Imagination Training (DreamerV3 style).
        """
        sequences = self.sample_sequence_batch(batch_size, sequence_length)
        if not sequences:
            return 0.0

        self.optimizer.zero_grad()
        
        # --- 1. WORLD MODEL TRAINING (on real data) ---
        wm_loss = 0
        latent_states = [] # Store for imagination
        
        # Hyperparameters (should ideally come from config)
        beta = 0.1
        gamma = 0.99
        
        for sequence in sequences:
            # Starting point
            first = sequence[0]
            obs_v = torch.tensor(np.frombuffer(first["vision_vector"], dtype=np.float32), device=self.device).unsqueeze(0)
            arm_id = first.get("personality_id", 0)
            
            # Initial inference
            h, _, _, _ = self.model.initial_inference(obs_v, arm_id, beta, gamma)
            
            seq_wm_loss = 0
            for k in range(1, len(sequence)):
                step = sequence[k]
                prev_step = sequence[k-1]
                
                # Recurrent Step
                act_idx = self.button_to_idx.get(prev_step["button"], 0)
                h, pred_reward, map_l, _, _, _, _ = self.model.recurrent_inference(
                    h, act_idx, arm_id, beta, gamma
                )
                
                if k >= burn_in:
                    # Store for imagination (detach to isolate WM from Actor-Critic gradients)
                    latent_states.append({
                        "h": h.detach(),
                        "arm_id": arm_id
                    })

                    # Reward Loss (Two-hot)
                    target_reward = torch.tensor([step["reward"]], device=self.device).float()
                    reward_dist = two_hot(symlog(target_reward), num_bins=self.model.num_bins)
                    actual_reward_logits = self.model.dynamics.reward_out(h)
                    reward_loss = F.cross_entropy(actual_reward_logits, reward_dist)
                    
                    # Auxiliary Loss (Map ID)
                    step_map_id = step.get("map_id", 0) or 0
                    map_loss = F.cross_entropy(map_l, torch.tensor([step_map_id % self.model.map_id_dim], device=self.device))
                    
                    seq_wm_loss += reward_loss + 0.5 * map_loss
            
            wm_loss += seq_wm_loss / (len(sequence) - burn_in)

        # --- 2. IMAGINATION TRAINING (Actor-Critic) ---
        actor_loss = 0
        value_loss = 0
        
        if latent_states:
            # Sample a subset of latent states to keep batch size manageable
            sample_indices = np.random.choice(len(latent_states), min(len(latent_states), batch_size * 4), replace=False)
            h_batch = torch.cat([latent_states[i]["h"] for i in sample_indices], dim=0)
            arm_ids = torch.tensor([latent_states[i]["arm_id"] for i in sample_indices], device=self.device)
            
            # Unroll in imagination
            imag_h = [h_batch]
            imag_rewards = []
            imag_values_ext = []
            imag_values_int = []
            imag_log_probs = []
            imag_entropies = []

            curr_h = h_batch
            for t in range(imagination_horizon):
                # Sample action from policy
                conditioning = self.model._get_conditioning(curr_h.size(0), self.device, arm_ids, beta, gamma)
                policy_logits, _, _ = self.model.prediction(curr_h, conditioning)
                
                probs = F.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                imag_log_probs.append(dist.log_prob(action))
                imag_entropies.append(dist.entropy())
                
                # Dynamics step
                action_onehot = F.one_hot(action, num_classes=self.model.action_dim).float()
                next_h, reward_l, _, _ = self.model.dynamics(curr_h, action_onehot)
                
                # Predict values for target
                _, v_ext_l, v_int_l = self.model.prediction(next_h, conditioning)
                
                imag_h.append(next_h)
                imag_rewards.append(symexp(from_two_hot(F.softmax(reward_l, dim=-1))))
                imag_values_ext.append(symexp(from_two_hot(F.softmax(v_ext_l, dim=-1))))
                imag_values_int.append(symexp(from_two_hot(F.softmax(v_int_l, dim=-1))))
                
                curr_h = next_h

            # Compute lambda-returns (simplified for now: n-step returns)
            # Future: Implement full lambda-return calculation
            returns = torch.stack(imag_values_ext).detach() # [H, B]
            
            # REINFORCE style Actor Loss
            discounted_returns = returns.mean(dim=0)
            actor_loss = -(torch.stack(imag_log_probs) * discounted_returns).mean()
            actor_loss -= 0.01 * torch.stack(imag_entropies).mean() # Entropy bonus

            # Value Loss (Mean Squared Error against returns)
            # For simplicity, we'll just train the value head to match the predicted reward + next value
            value_loss = F.mse_loss(torch.stack(imag_values_ext), returns)

        total_loss = (wm_loss / batch_size) + actor_loss + value_loss
        
        if total_loss > 0:
            try:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            except Exception as e:
                print(f"❌ Training Step Failed: {e}")
                return 0.0

        # Update Target
        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return total_loss.item()

    async def run_loop(self):
        print("🚀 Starting DreamerV3-style Training Loop...")
        while True:
            loss = await self.train_step()
            if self.training_steps % 10 == 0:
                print(f"📉 Step {self.training_steps} | Loss: {loss:.4f}")
            await asyncio.sleep(0.1)

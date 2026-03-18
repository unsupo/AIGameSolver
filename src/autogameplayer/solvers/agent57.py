import torch
import random
from typing import List, Dict, Optional, Any, Tuple

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry
from autogameplayer.core.config import settings
from autogameplayer.core.bandit import SlidingWindowUCB, PolicyArm
from autogameplayer.muzero.networks import MuZeroModel
from autogameplayer.vision.encoder import VisionEncoder


@SolverRegistry.register("agent57")
class Agent57Solver(BaseSolver):
    """
    Agent57 Solver: Multi-armed bandit selects (gamma, beta) arms.
    Uses a recurrent MuZeroModel with UVFA conditioning.
    """

    def __init__(
        self,
        supported_buttons: List[str],
        input_dim=None,
        hidden_dim=256,
        num_arms: int = 32,
        model_name="agent57_weights.pth",
        **kwargs,
    ):
        self.buttons = supported_buttons
        self.action_dim = len(self.buttons)
        self.input_dim = input_dim or VisionEncoder.get_dim(settings.vision_model)
        
        # 1. Initialize Model
        self.model = MuZeroModel(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            action_dim=self.action_dim,
            num_personalities=num_arms
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device)
        
        # 2. Meta-Controller (Bandit)
        db_path = settings.models_dir / "bandit_stats.db"
        self.num_arms = num_arms
        # Using a simple sync init for now, would be async in a real factory
        self.bandit = SlidingWindowUCB(num_arms=num_arms, db_path=db_path)
        
        # 3. Active Arm State
        self.current_arm: Optional[PolicyArm] = None
        self.current_hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        self.save_path = settings.models_dir / model_name
        self._load_weights()

    def _load_weights(self):
        if self.save_path.exists():
            try:
                self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
            except Exception:
                pass

    async def reset(self):
        # Episode start: Select new arm
        self.current_arm = await self.bandit.select_arm()
        self.current_hidden_state = None
        print(f"🎯 Agent57: Selected Arm {self.current_arm.id} (β={self.current_arm.beta:.3f}, γ={self.current_arm.gamma:.3f}, ε={self.current_arm.epsilon:.3f})")

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        if self.current_arm is None:
            await self.reset()
            
        # 1. Epsilon-Greedy (Bandit-driven)
        if random.random() < self.current_arm.epsilon:
            action_idx = random.randrange(self.action_dim)
            reasoning = f"Arm {self.current_arm.id} Exploration (ε={self.current_arm.epsilon:.2f})"
            confidence = self.current_arm.epsilon
        else:
            # 2. Model Inference
            obs_tensor = torch.tensor(obs.state.vision_vector, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                if self.current_hidden_state is None:
                    # Initial step in episode
                    self.current_hidden_state, logits, v_e, v_i = self.model.initial_inference(
                        obs_tensor, 
                        personality_id=self.current_arm.id,
                        beta=self.current_arm.beta,
                        gamma=self.current_arm.gamma
                    )
                else:
                    # Recurrent step would usually happen AFTER previous action, 
                    # but MuZero architecture allows initial_inference on every 'real' observation
                    # to keep hidden state grounded in reality.
                    # Alternatively, we could unroll.
                    self.current_hidden_state, logits, v_e, v_i = self.model.initial_inference(
                        obs_tensor, 
                        personality_id=self.current_arm.id,
                        beta=self.current_arm.beta,
                        gamma=self.current_arm.gamma
                    )
                
                # Q = Q_e + beta * Q_i
                # In this simplified version, we use policy logits directly
                action_idx = torch.argmax(logits).item()
                confidence = float(torch.softmax(logits, dim=-1).max())
                reasoning = f"Arm {self.current_arm.id} Policy Prediction"

        # Update context metadata
        obs.state.context["agent57_arm"] = self.current_arm.id
        
        button = self.buttons[action_idx]
        return ActionProposal(
            action=Action(button=button, reasoning=reasoning),
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "arm_id": self.current_arm.id,
                "logits": logits.tolist() if "logits" in locals() else None,
                "solver_name": "Agent57Solver",
            },
        )

    async def on_episode_end(self, total_reward: float):
        if self.current_arm:
            await self.bandit.report_episode_result(self.current_arm.id, total_reward)
        
        # Save weights
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Agent57Solver",
            "num_arms": self.num_arms,
            "active_arm": self.current_arm.id if self.current_arm else None,
            "device": str(self.device)
        }

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from autogameplayer.muzero.networks import MuZeroModel


class MentalSimulator:
    """
    MuZero Dynamics-based Simulator.
    Allows the agent to 'imagine' future sequences of actions without
    running the emulator.
    """

    def __init__(self, model: MuZeroModel):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Mapping for output
        from autogameplayer.core.controllers import STANDARD_BUTTONS

        self.buttons = STANDARD_BUTTONS

    def imagine_path(
        self, initial_hidden_state: torch.Tensor, action_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Projects a specific sequence of actions into the future.
        Returns a list of (predicted_reward, predicted_value).
        """
        results = []
        current_h = initial_hidden_state
        if current_h.dim() == 1:
            current_h = current_h.unsqueeze(0)

        with torch.no_grad():
            for action_idx in action_indices:
                # g(s, a) -> next_s, r, map_l, ocr_l; phi(next_s) -> p, v
                next_h, reward, map_l, ocr_l, policy_logits, value = (
                    self.model.recurrent_inference(current_h, action_idx)
                )

                results.append(
                    {
                        "action": self.buttons[action_idx]
                        if action_idx < len(self.buttons)
                        else "unknown",
                        "predicted_reward": float(reward.item()),
                        "predicted_value": float(value.item()),
                        "policy_dist": F.softmax(policy_logits, dim=-1)
                        .cpu()
                        .numpy()[0]
                        .tolist(),
                    }
                )
                current_h = next_h

        return results

    def find_best_mental_sequence(
        self,
        initial_hidden_state: torch.Tensor,
        depth: int = 5,
        num_sequences: int = 20,
    ) -> Tuple[List[int], float]:
        """
        Randomly samples action sequences and 'imagines' their outcomes.
        Returns the sequence with the highest cumulative reward + final value.
        """
        best_score = -float("inf")
        best_sequence = []

        action_dim = self.model.action_dim

        for _ in range(num_sequences):
            # Generate a random sequence
            sequence = [np.random.randint(0, action_dim) for _ in range(depth)]

            # Imagine it
            path_results = self.imagine_path(initial_hidden_state, sequence)

            # Score = sum of rewards + discounted final value
            score = sum(r["predicted_reward"] for r in path_results)
            score += 0.99 * path_results[-1]["predicted_value"]

            if score > best_score:
                best_score = score
                best_sequence = sequence

        return best_sequence, best_score

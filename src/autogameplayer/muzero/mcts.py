import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from autogameplayer.muzero.networks import MuZeroModel


class MCTSNode:
    """A node in the MuZero MCTS search tree."""

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.children: Dict[int, MCTSNode] = {}
        self.hidden_state: Optional[torch.Tensor] = None
        self.reward = 0.0

    @property
    def value(self) -> float:
        """Average value of the node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_action(self, pb_c_base: float, pb_c_init: float) -> int:
        """Selects the action with the highest PUCT score."""
        scores = {}
        for action, child in self.children.items():
            scores[action] = self.get_puct_score(child, pb_c_base, pb_c_init)

        return max(scores, key=scores.get)

    def get_puct_score(
        self, child: "MCTSNode", pb_c_base: float, pb_c_init: float
    ) -> float:
        """Polynomial Upper Confidence Tree score."""
        pb_c = math.log((self.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= math.sqrt(self.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value
        return prior_score + value_score


class MCTS:
    """Monte Carlo Tree Search implementation for MuZero."""

    def __init__(
        self,
        model: MuZeroModel,
        num_simulations: int = 50,
        pb_c_base: float = 19652,
        pb_c_init: float = 1.25,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = 0.99

    def search(
        self,
        observation: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Performs MCTS search starting from either a physical observation or a persistent latent state.
        """
        with torch.no_grad():
            if initial_state is not None:
                # Start from the persistent recurrent state
                hidden_state = initial_state
                # Ensure batch dimension
                if hidden_state.dim() == 1:
                    hidden_state = hidden_state.unsqueeze(0)
                policy_logits, value = self.model.prediction(hidden_state)
            elif observation is not None:
                # Start from a fresh observation (Markovian fallback)
                hidden_state, policy_logits, value = self.model.initial_inference(
                    observation.unsqueeze(0)
                )
            else:
                raise ValueError("Must provide either observation or initial_state")

            probs = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]

            root = MCTSNode(0)
            root.hidden_state = hidden_state
            self.expand_node(root, probs)

            for _ in range(self.num_simulations):
                node = root
                search_path = [node]

                # 1. Selection
                while node.children and any(
                    c.visit_count > 0 for c in node.children.values()
                ):
                    action = node.select_action(self.pb_c_base, self.pb_c_init)
                    node = node.children[action]
                    search_path.append(node)

                    # If we hit a node that hasn't been evaluated, stop selection
                    if node.hidden_state is None:
                        break

                # 2. Expansion and Evaluation
                parent = search_path[-2] if len(search_path) > 1 else None
                if parent:
                    # Find which action led to this child
                    action = [a for a, c in parent.children.items() if c == node][0]
                    # g(s, a) -> next_s, r, map_l, ocr_l; phi(next_s) -> p, v
                    next_s, reward, map_l, ocr_l, p_logits, v = (
                        self.model.recurrent_inference(parent.hidden_state, action)
                    )

                    node.hidden_state = next_s
                    node.reward = reward.item()
                    probs = F.softmax(p_logits, dim=-1).cpu().numpy()[0]
                    self.expand_node(node, probs)
                    evaluation = v.item()
                else:
                    # Root already expanded
                    evaluation = value.item()

                # 3. Backpropagation
                self.backpropagate(search_path, evaluation)

        # Return best action (most visited)
        visit_counts = np.array(
            [root.children[a].visit_count for a in range(self.model.action_dim)]
        )
        best_action = np.argmax(visit_counts)

        # Action probabilities for training (policy target)
        action_probs = visit_counts / np.sum(visit_counts)

        return int(best_action), action_probs

    def expand_node(self, node: MCTSNode, probs: np.ndarray):
        """Expands a node by creating children for all possible actions."""
        for action, prob in enumerate(probs):
            node.children[action] = MCTSNode(prob)

    def backpropagate(self, search_path: List[MCTSNode], value: float):
        """Updates values and visit counts along the search path."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            # Add reward to value for parent nodes
            value = node.reward + self.discount * value

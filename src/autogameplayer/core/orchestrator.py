import asyncio
import uuid
from typing import Optional

from autogameplayer.core.environment import EmulatorEnvironment
from autogameplayer.core.solver import BaseSolver
from autogameplayer.core.context import AgentContext
from autogameplayer.core.models import Observation, Action
from autogameplayer.utils.telemetry import telemetry


class SessionOrchestrator:
    """
    Manages the session lifecycle, delegating decisions to the active Solver pipeline
    instead of the legacy monolithic Brain.
    """

    def __init__(
        self,
        env: EmulatorEnvironment,
        solver: BaseSolver,
        context: AgentContext,
        render_delay: float = 0.05,
    ):
        self.env = env
        self.solver = solver
        self.context = context
        self.render_delay = render_delay
        self.session_id = str(uuid.uuid4())[:8]
        self.total_reward = 0.0
        self.last_obs: Optional[Observation] = None

    async def hot_swap_solver(self, new_solver: BaseSolver):
        """Replaces the current solver while preserving state if possible."""
        print(f"🔄 Hot-swapping solver: {self.solver.__class__.__name__} -> {new_solver.__class__.__name__}")
        new_solver.transfer_state(self.solver)
        self.solver = new_solver
        # No need to reset() here as we want to keep the current arm/state if it transferred

    async def _handle_map_transition(self, current_map_id: int, last_map_id: int):
        pass

    async def run(self, steps: int = 100) -> float:
        """Executes the game loop using the solver."""
        print(f"▶️ Starting Session Orchestrator {self.session_id}...")

        obs = await self.env.reset()
        last_map_id = obs.state.context.get("map_id", -1)
        self.context.metadata["is_dialogue"] = obs.state.has_dialogue_box

        try:
            for i in range(1, steps + 1):
                # 1. Ask Solver for Proposal
                proposal = await self.solver.propose_action(obs, self.context)

                # Telemetry
                solver_name = (
                    proposal.source
                    or proposal.metadata.get("solver_name", "UnknownSolver")
                    if proposal.metadata
                    else (proposal.source or "UnknownSolver")
                )
                if "macro_name" in proposal.metadata:
                    solver_name = "MacroSolver"
                elif "mcts_probs" in proposal.metadata:
                    solver_name = "MCTSSolver"

                telemetry.log_decision(
                    solver_name, proposal.confidence, proposal.reasoning
                )

                # 2. Step Environment
                next_obs, reward, done = await self.env.step(proposal.action)

                self.total_reward += reward
                current_map_id = next_obs.state.context.get("map_id", -1)
                next_coords = (
                    next_obs.state.context.get("x", 0),
                    next_obs.state.context.get("y", 0),
                )

                # --- AGENT CONTEXT UPDATES (Stagnation & Loops) ---
                # Odometer Collision Detection
                action_btn = (
                    proposal.action.button.lower()
                    if isinstance(proposal.action, Action) and proposal.action.button
                    else (proposal.action.lower() if isinstance(proposal.action, str) else "")
                )
                if action_btn in ["up", "down", "left", "right"]:
                    if self.context.coords == next_coords and current_map_id == self.context.map_id:
                        # Coordinates didn't change despite movement action -> Collision
                        # Calculate the wall's coordinates based on direction
                        wx, wy = next_coords
                        if action_btn == "up":
                            wy -= 1
                        elif action_btn == "down":
                            wy += 1
                        elif action_btn == "left":
                            wx -= 1
                        elif action_btn == "right":
                            wx += 1
                        
                        collision = (wx, wy)
                        if collision not in self.context.collision_history:
                            print(f"🚧 Odometer: Collision at {collision}. This tile is BLOCKED.")
                            self.context.collision_history.append(collision)
                            if len(self.context.collision_history) > 10:
                                self.context.collision_history.pop(0)

                self.context.last_reward = reward
                self.context.map_id = current_map_id
                self.context.coords = next_coords

                # Track Vision Stagnation
                if next_obs.state.vision_delta < 0.05:
                    self.context.consecutive_stuck_steps += 1
                else:
                    self.context.consecutive_stuck_steps = 0

                self.context.is_stuck = self.context.consecutive_stuck_steps >= 20

                # Simple Loop Detection (Rolling hash window)
                if "hash_history" not in self.context.metadata:
                    self.context.metadata["hash_history"] = []
                self.context.metadata["hash_history"].append(next_obs.state_hash)
                if len(self.context.metadata["hash_history"]) > 20:
                    self.context.metadata["hash_history"].pop(0)

                self.context.is_loop_detected = (
                    self.context.metadata["hash_history"].count(next_obs.state_hash) > 3
                )
                # --------------------------------------------------

                # Handlers
                await self._handle_map_transition(current_map_id, last_map_id)
                last_map_id = current_map_id

                # Update Context with new metadata
                self.context.metadata["is_dialogue"] = next_obs.state.has_dialogue_box

                # --- PHASE 3: Solver Feedback Loop ---
                outcome = {
                    "reward": reward,
                    "done": done,
                    "is_stuck": self.context.is_stuck,
                    "is_loop_detected": self.context.is_loop_detected,
                    "next_obs": next_obs
                }
                await self.solver.on_step_end(proposal, reward, done, outcome=outcome)
                # ------------------------------------

                obs = next_obs
                self.last_obs = obs

                if done:
                    break

                if self.render_delay > 0:
                    await asyncio.sleep(self.render_delay)

        except asyncio.CancelledError, KeyboardInterrupt:
            print(f"Session {self.session_id} cancelled.")

        # Post-Session Breakdown
        stats = telemetry.get_session_summary()
        print(
            f"🌙 Session End. Steps: {stats['total_steps']} | Distribution: {stats['solver_usage']}"
        )

        await self.solver.reset()
        return self.total_reward

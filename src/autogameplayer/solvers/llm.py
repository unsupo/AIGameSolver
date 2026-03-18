import random
from typing import Dict, Any, List, Optional

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry
from autogameplayer.utils.llm import LLMClientProtocol, extract_json_from_llm_response


@SolverRegistry.register("llm")
class LLMSolver(BaseSolver):
    """
    A solver that uses a Large Language Model to decide the next action or sequence of actions.
    Specifically designed to break through bottlenecks and stagnation.
    """

    def __init__(
        self,
        client: LLMClientProtocol,
        model: str,
        buttons: List[str],
        system_prompt: Optional[str] = None,
    ):
        self.client = client
        self.model = model
        self.buttons = buttons
        self.system_prompt = system_prompt or "You are an expert Pokémon player and reinforcement learning meta-reasoner."

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        print(f"🧠 LLM Solver: Analyzing bottleneck at {context.map_id} ({context.coords})...")
        
        # Build a detailed prompt for the LLM to understand the bottleneck
        history = "\n".join(context.recent_actions[-10:])
        collisions = ", ".join([f"({x},{y})" for x, y in context.collision_history])
        collision_text = f"\nRECENT COLLISIONS (Walls): {collisions}" if collisions else ""
        
        prompt = f"""
{self.system_prompt}

The agent is currently STUCK or in a LOOP.
Break out of this bottleneck by providing a multi-step macro sequence.
You MUST output a sequence of 3 to 5 actions (a 'macro') to aggressively break out of this state.
Examples: 'B, B, B, LEFT' to close a menu and walk away, or 'DOWN, DOWN, RIGHT, A' to navigate around a corner.
{collision_text}

CURRENT STATE:
...
- Map: {context.map_id}
- Coordinates: {context.coords}
- Vision Delta: {obs.state.vision_delta:.4f}
- Stagnation Counter: {context.consecutive_stuck_steps}
- Is Loop Detected: {context.is_loop_detected}
- OCR Text: "{obs.state.ocr_text}"

RECENT HISTORY:
{history}

GOAL: Break out of this bottleneck. Provide a COHERENT SEQUENCE of actions (a macro) to move the agent to a new state.
If you are in the overworld, try to walk AROUND any recent collisions.
If you are in a dialogue, press 'A' or 'B' to advance.

IMPORTANT: Respond ONLY with a JSON object.
Example:
{{
  "macro": [
    {{"button": "right", "frames": 20}},
    {{"button": "up", "frames": 30}},
    {{"button": "up", "frames": 30}}
  ],
  "reasoning": "Walking right to clear the obstacle at {context.coords} before heading North."
}}
"""

        try:
            content = [{"type": "text", "text": prompt}]
            if obs.state.image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{obs.state.image_data}"}
                })

            response_text = await self.client.acreate_completion(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=300,
                temperature=0.0,
            )

            result = extract_json_from_llm_response(response_text)
            if not result or "macro" not in result:
                # Fallback to single action if macro not provided
                btn = result.get("button", random.choice(self.buttons)) if result else random.choice(self.buttons)
                action = Action(button=btn, reasoning=result.get("reasoning", "LLM Fallback") if result else "LLM Parse Failure")
                return ActionProposal(action=action, confidence=0.5, reasoning=action.reasoning)

            macro_actions = [
                Action(button=step["button"], duration=step.get("frames", 10))
                for step in result["macro"]
            ]
            
            # Wrap in a top-level Action object
            macro_action = Action(
                macro=macro_actions,
                reasoning=result.get("reasoning", "LLM Bottleneck Breakout")
            )

            return ActionProposal(
                action=macro_action,
                confidence=1.0,
                reasoning=macro_action.reasoning,
                metadata={"macro_sequence": macro_actions, "solver_name": "LLMSolver"}
            )

        except Exception as e:
            print(f"⚠️ LLM Solver Error: {e}")
            fallback_btn = random.choice(["up", "a", "b"])
            return ActionProposal(
                action=Action(button=fallback_btn, reasoning=f"Error fallback: {e}"),
                confidence=0.1,
                reasoning=f"Error in LLM Solver: {e}",
                metadata={"solver_name": "LLMSolver"}
            )

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "model": self.model,
            "type": "reasoner"
        }

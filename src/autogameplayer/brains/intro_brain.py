import time
from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.registry import Registry
from autogameplayer.core.optimizer import StrategyOptimizer
from autogameplayer.brains.agentic.memory import LongTermMemory, EpisodicMemory
from autogameplayer.utils.llm import LLMClientProtocol

from autogameplayer.brains.agentic.reflector import ReflectionAgent


@Registry.register_brain("intro")
class IntroMashBrain(Brain):
    """A brain designed to get past intro screens using learned macros or structured patterns."""

    def __init__(
        self,
        controller: Controller,
        optimizer: StrategyOptimizer = None,
        llm_client: LLMClientProtocol = None,
        **kwargs,
    ):
        self.controller = controller

        if llm_client is None:
            from autogameplayer.utils.llm import get_llm_client

            self.client = get_llm_client()
        else:
            self.client = llm_client

        # Unified Memory Infrastructure
        self.optimizer = optimizer or StrategyOptimizer(self.client)
        self.long_term_memory = LongTermMemory(self.client)
        self.session_id = f"intro_{int(time.time())}"
        self.step_count = 0
        self.memory = EpisodicMemory()  # Added for step recording
        self.reflector = ReflectionAgent(
            self.client, "nomic-embed-text", self.optimizer
        )

    @property
    def has_reflection(self) -> bool:
        return True

    async def act(self, observation: Observation, mcp_client=None) -> Action:
        # 1. Check for a learned intro macro first (Confidence > 0.7)
        if self.optimizer and observation.state.vision_vector:
            map_id = observation.state.context.get("map_id", 0)
            macros = self.optimizer.get_best_macro_for_context(
                vision_vector=observation.state.vision_vector, map_id=map_id, top_k=1
            )
            if macros:
                best = macros[0]
                if best.get("reliability", 1.0) > 0.7:
                    print(
                        f"⚡ Intro Shortcut: {best['description']} (Reliability: {best.get('reliability', 1.0):.2f})"
                    )
                    macro_actions = [Action(**step) for step in best["sequence"]]
                    action = Action(
                        macro=macro_actions,
                        reasoning=f"Learned intro shortcut: {best['description']}",
                    )
                    self.memory.record_step(observation, action)
                    self.step_count += 1
                    return action

        # 2. Fallback: Structured pattern instead of pure random
        # Title screens universally respond to: wait, A, wait, A pattern
        INTRO_PATTERN = [
            {"button": "a", "frames": 10},
            {"button": "a", "frames": 20},
            {"button": "start", "frames": 10},
            {"button": "a", "frames": 15},
        ]

        step = self.step_count % len(INTRO_PATTERN)
        self.step_count += 1
        p = INTRO_PATTERN[step]

        reasoning = (
            f"Executing structured intro pattern step {step + 1}: {p['button'].upper()}"
        )
        action = Action(button=p["button"], duration=p["frames"], reasoning=reasoning)
        self.memory.record_step(observation, action)
        return action

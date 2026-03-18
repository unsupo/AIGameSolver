import random
import asyncio
import time
from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.config import settings
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.utils.llm import (
    LLMClientProtocol,
    OllamaBootstrap,
    extract_json_from_llm_response,
)
from autogameplayer.core.registry import Registry

from autogameplayer.core.optimizer import StrategyOptimizer
from autogameplayer.core.knowledge import KnowledgeBase
from autogameplayer.brains.agentic.reflector import ReflectionAgent

from .agentic.memory import EpisodicMemory, LongTermMemory
from .agentic.critic import CriticAgent

from .agentic.actor import ActorAgent


@Registry.register_brain("llm")
class LLMBrain(Brain):
    """A high-intelligence LLM brain capable of navigating intros and naming screens."""

    def __init__(
        self,
        controller: Controller,
        config: GameConfig = None,
        llm_client: LLMClientProtocol = None,
    ):
        self._tasks: set[asyncio.Task] = set()
        self.controller = controller
        self.config = config
        self.buttons = self.controller.buttons
        self.history = []
        self.summary = "Just started the game."
        self.last_outcome = "Fresh start."
        self.model = (config.llm_model if config else None) or settings.llm_model
        self.session_id = f"session_{int(time.time())}"  # Unique ID for this run
        self.step_count = 0
        self.last_map_id = -1
        self._is_reflecting = False

        # Bootstrap handled by utility
        OllamaBootstrap.bootstrap([self.model, "nomic-embed-text"])

        if llm_client is None:
            from autogameplayer.utils.llm import get_llm_client

            self.client = get_llm_client()
        else:
            self.client = llm_client

        # Unified Memory Infrastructure
        self.memory = EpisodicMemory()
        self.long_term_memory = LongTermMemory(self.client)
        self.optimizer = StrategyOptimizer(self.client, self.model)
        self.critic = CriticAgent(
            ltm=self.long_term_memory, session_id=self.session_id, config=self.config
        )
        self.knowledge = KnowledgeBase(self.client)
        self.actor = ActorAgent(
            self.client,
            self.model,
            self.buttons,
            self.config,
            ltm=self.long_term_memory,
            optimizer=self.optimizer,
        )
        self.reflector = ReflectionAgent(self.client, self.model, self.optimizer)

        print(f"🧠 LLM Brain Initialized | Model: {self.model}")

    @property
    def has_reflection(self) -> bool:
        return True

    async def act(self, observation: Observation, mcp_client=None) -> Action:
        # 1. Process PREVIOUS step outcome
        await self._process_step_outcome(observation, mcp_client=mcp_client)

        ctx = observation.state.context
        map_id = ctx.get("map_id", 0)

        # 2. Periodic Optimization
        if self.step_count % 50 == 0:
            self.optimizer.optimize()
            task = asyncio.create_task(self.optimizer.async_compress_skills())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        # NEW: Macro Evolution (Every 500 steps)
        if self.step_count % 500 == 0 and self.step_count > 0:
            print("🧬 AI is 'Evolving' (Genetic Algorithm on Macro population)...")
            self.optimizer.evolve_population(top_k=5)

        # 3. Agnostic Auto-Pilot Heuristic
        auto_pilot_threshold = (
            self.config.heuristics.auto_pilot_until_map if self.config else 0
        )
        if map_id < auto_pilot_threshold:
            action = self._handle_auto_pilot(map_id)
            self.memory.record_step(observation, action)
            self.step_count += 1
            return action

        # 4. Use Actor Orchestrator (Policy -> Specialized Agents)
        try:
            # We use the current summary as the plan for the actor
            action = await self.actor.get_next_action(
                observation, self.summary, self.memory, mcp_client=mcp_client
            )
            # Update internal state from action reasoning
            if action.button:
                self.history.append(action.button.upper())
        except Exception as e:
            print(f"⚠️ LLM Action Error: {e}")
            action = self.fallback_action(self.controller)

        self.memory.record_step(observation, action)
        self.step_count += 1
        return action

    async def _process_step_outcome(self, observation: Observation, mcp_client=None):
        reward_delta, is_stuck, _, _ = self.critic.evaluate(self.memory, observation)
        self.memory.update_last_step(observation, is_stuck)

        if self.memory.steps:
            last_record = self.memory.steps[-1]
            task = asyncio.create_task(
                self.long_term_memory.add_step_to_replay(
                    self.session_id, self.step_count - 1, last_record
                )
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self.last_map_id = observation.state.context.get("map_id", -1)

    async def close(self):
        if self._tasks:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _handle_auto_pilot(self, map_id: int) -> Action:
        if random.random() < 0.8:
            btn = random.choice(["a", "start"])
        else:
            btn = random.choice(["down", "right"])
        print(f"🤖 Auto-Pilot (Map #{map_id}): Pressing {btn.upper()}...")
        return Action(
            button=btn,
            duration=15,
            reasoning=f"Auto-piloting through intro phase (Map {map_id}).",
        )

    def _build_prompt(
        self, ctx: dict, observation: Observation, memories: str, external: str
    ) -> str:
        history_text = " -> ".join(self.history[-10:]) if self.history else "None"
        map_id = ctx.get("map_id", 0)
        game_desc = self.config.description if self.config else "this game"

        intro_guidance = ""
        if map_id == 0 and self.config and self.config.profile:
            intro_guidance = self.config.profile.intro_guidance

        skill_context = self.optimizer.get_skills_for_map(map_id)

        ram_text = f"PLAYER POSITION: ({ctx.get('x')}, {ctx.get('y')}) on Map #{map_id} | HP: {ctx.get('hp')}"

        return f"""
        You are an expert player of {game_desc}.
        {ram_text}
        {intro_guidance}
        {skill_context}
        {external}
        
        STRATEGY: {self.summary}
        HISTORY: {history_text}
        RECALLED MEMORIES:
        {memories}
        
        TOOLS AVAILABLE:
        - scan_memory: Read a RAM address. Use to check player HP, badge count, map progress.
        - save_state / load_state: Save your progress (use after clearing a room or before a risky action).
          Save slot {settings.bootstrap_slot} = bootstrap (don't overwrite). Use slots {settings.rolling_save_start}-{settings.rolling_save_end} for branching exploration.

        KEY POKEMON RED ADDRESSES:
        - 0xD35E = Current Map ID
        - 0xD362 = Player X tile  
        - 0xD361 = Player Y tile
        - 0xD16B = Badge count (0-8)
        - 0xD163 = Party size
        - 0xD31C = Player money (3 bytes BCD)
        
        OCR: "{observation.state.ocr_text or "No text"}"
        
        TASK: Decide the next button press to advance the game.
        
        RESPONSE (JSON ONLY):
        {{
            "button": "name",
            "reasoning": "...",
            "goal_immediate": "..."
        }}
        """

    async def _get_llm_action(
        self, prompt: str, observation: Observation, map_id: int
    ) -> Action:
        print(f"⏳ AI Thinking (Map #{map_id})...")

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{observation.state.image_data}"
                },
            },
        ]

        response_text = await self.client.acreate_completion(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=200,
            temperature=0.1,
            timeout=30.0,
        )

        result = extract_json_from_llm_response(response_text)
        if not result:
            raise ValueError(f"Failed to parse LLM response: {response_text}")

        button = str(result.get("button", "a")).lower().strip()
        reasoning = result.get("reasoning", "Thinking...")

        # Update Internal State
        self.history.append(button.upper())
        self.summary = result.get("goal_immediate", "Progressing")
        self.last_outcome = f"Pressed {button.upper()}. {reasoning}"

        print(f"🤖 AI Action: {button.upper().ljust(6)} | Goal: {self.summary[:50]}")

        duration = (
            15 if "text" in reasoning.lower() or "dialog" in reasoning.lower() else 5
        )
        return Action(button=button, duration=duration, reasoning=reasoning)

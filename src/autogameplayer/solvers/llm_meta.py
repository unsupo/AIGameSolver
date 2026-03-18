import asyncio
import json
from typing import Dict, Any, List, Optional

from autogameplayer.core.models import Observation, Action
from autogameplayer.core.solver import BaseSolver, ActionProposal
from autogameplayer.core.context import AgentContext
from autogameplayer.solvers.registry import SolverRegistry
from autogameplayer.utils.llm import LLMClientProtocol, extract_json_from_llm_response
from autogameplayer.core.knowledge import KnowledgeBase
from autogameplayer.core.config import settings
from autogameplayer.brains.agentic.reflector import ReflectionAgent


@SolverRegistry.register("llm_meta")
class LLMMetaSolver(BaseSolver):
    """
    Symbolic Meta-Solver that synthesizes ActionProposals from multiple advisors.
    Uses confidence gating, RAG, and budget tracking.
    """

    def __init__(
        self,
        client: LLMClientProtocol,
        model: str,
        supported_buttons: List[str],
        advisors: Dict[str, BaseSolver],
        knowledge_base: Optional[KnowledgeBase] = None,
        reflector: Optional[ReflectionAgent] = None,
        invoke_threshold: float = 0.4,
        timeout: float = 15.0,
        max_calls_per_episode: int = 5,
        **kwargs,
    ):
        self.client = client
        self.model = model
        self.buttons = supported_buttons
        self.advisors = advisors
        self.kb = knowledge_base
        self.reflector = reflector
        self.invoke_threshold = invoke_threshold
        self.timeout = timeout
        self.max_calls = max_calls_per_episode
        
        self.calls_this_episode = 0
        self.total_tokens_used = 0
        self.response_cache = {} # (state_hash, plan_hash) -> proposal
        self.cost_log_path = settings.base_dir / "logs" / "llm_costs.json"

    async def reset(self):
        self.calls_this_episode = 0
        for advisor in self.advisors.values():
            await advisor.reset()

    async def propose_action(
        self, obs: Observation, context: AgentContext
    ) -> ActionProposal:
        # 1. Cache Lookup
        goal_hash = hash(context.current_goal) if context.current_goal else 0
        cache_key = (obs.state_hash, goal_hash)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        # 2. Gather Advisor Proposals
        advisor_proposals = {}
        best_advisor_name = None
        max_confidence = -1.0

        for name, advisor in self.advisors.items():
            prop = await advisor.propose_action(obs, context)
            advisor_proposals[name] = prop
            if prop.confidence > max_confidence:
                max_confidence = prop.confidence
                best_advisor_name = name

        # 3. Confidence Gating
        force_reasoning = context.is_stuck or context.is_loop_detected
        
        if (max_confidence >= self.invoke_threshold and not force_reasoning) or \
           (self.calls_this_episode >= self.max_calls):
            return advisor_proposals[best_advisor_name]

        # 4. LLM Meta-Reasoning
        print(f"🤔 LLM Meta-Reasoner: Confidence low ({max_confidence:.2f}). Invoking model {self.model}...")
        self.calls_this_episode += 1
        
        try:
            # RAG Context
            rag_snippets = []
            if self.kb:
                query = f"Map: {context.map_id}, Goal: {context.current_goal}, OCR: {obs.state.ocr_text}"
                rag_snippets = await self.kb.query(query, top_k=2)

            proposal = await asyncio.wait_for(
                self._invoke_llm(obs, context, advisor_proposals, rag_snippets),
                timeout=self.timeout
            )
            
            # Update cache
            self.response_cache[cache_key] = proposal
            return proposal

        except asyncio.TimeoutError:
            print("⏳ LLM Meta-Reasoner: Timeout! Falling back to best advisor.")
            return advisor_proposals[best_advisor_name]
        except Exception as e:
            print(f"⚠️ LLM Meta-Reasoner Error: {e}")
            return advisor_proposals[best_advisor_name]

    async def _invoke_llm(
        self, 
        obs: Observation, 
        context: AgentContext, 
        advisors: Dict[str, ActionProposal],
        rag_snippets: List[str]
    ) -> ActionProposal:
        # 1. Build Structured Perception Scaffolding
        scaffolding = f"""
### STRUCTURED PERCEPTION PAYLOAD:
- OCR text: "{obs.state.ocr_text or 'None detected'}"
- SLAM state: map_id={context.map_id}, coords={context.coords}, explored_tiles={len(context.visited_locations)}
- RAM probes: { {k: v for k, v in context.raw.items() if isinstance(v, (int, float, str))} }
- Top solver proposals:
"""
        for name, p in advisors.items():
            scaffolding += f"  * {name} (Conf: {p.confidence:.2f}): {p.reasoning}. Action: {p.action}\n"
        
        scaffolding += f"- Active Goal: {context.current_goal or 'Explore and advance.'}\n"
        scaffolding += f"- Recent Reward Trend: {context.last_reward:.2f}\n"

        # 2. Build Knowledge Text
        kb_text = "\n".join([f"- {s}" for s in rag_snippets]) if rag_snippets else "No specific knowledge found."

        # Prompt
        prompt = f"""
You are the High-Level Executive for an autonomous game-playing agent.
Your task is to synthesize structured perception data, advisor suggestions, and environmental context to decide the best action.

{scaffolding}

RELEVANT KNOWLEDGE (RAG):
{kb_text}

REQUIREMENTS:
1. If the agent is stuck, provide a macro sequence (3-5 steps) to break out.
2. If multiple advisors disagree, weight their confidence and explain your choice.
3. Align with the knowledge snippets if applicable.
4. Respond ONLY with a valid JSON object.

Example:
{{
  "action": "up",
  "confidence": 0.9,
  "reasoning": "Moving North to follow the walkthrough guidance for the next milestone.",
  "macro": [
    {{"button": "b", "duration": 5}},
    {{"button": "left", "duration": 20}}
  ]
}}
"""

        content = [{"type": "text", "text": prompt}]
        if obs.state.image_data:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{obs.state.image_data}"}
            })

        response_text = await self.client.acreate_completion(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=400,
            temperature=0.0
        )

        result = extract_json_from_llm_response(response_text)
        if not result:
            raise ValueError("Failed to parse LLM response.")

        # Construct Action
        macro = None
        if "macro" in result:
            macro = [Action(button=step["button"], duration=step.get("duration", 5)) for step in result["macro"]]
        
        final_action = Action(
            button=result.get("action"),
            macro=macro,
            reasoning=result.get("reasoning", "LLM Meta-Decision")
        )

        # Track usage (placeholder for true token counting)
        self.total_tokens_used += len(prompt.split()) + 1000 # Estimate for image
        self._log_cost()

        return ActionProposal(
            action=final_action,
            confidence=float(result.get("confidence", 1.0)),
            reasoning=final_action.reasoning,
            source="llm_meta",
            metadata={"llm_response": result, "solver_name": "LLMMetaSolver"}
        )

    def _log_cost(self):
        try:
            self.cost_log_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"total_tokens": self.total_tokens_used, "calls": self.calls_this_episode}
            with open(self.cost_log_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    async def on_step_end(self, proposal: ActionProposal, reward: float, done: bool, outcome: dict = None):
        for advisor in self.advisors.values():
            await advisor.on_step_end(proposal, reward, done, outcome=outcome)

    async def on_episode_end(self, total_reward: float):
        for advisor in self.advisors.values():
            await advisor.on_episode_end(total_reward)
        
        # Reflection logic integration would go here if needed
        # self.reflector.analyze_session(...)

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "LLMMetaSolver",
            "model": self.model,
            "advisors": list(self.advisors.keys()),
            "rag_enabled": self.kb is not None,
            "token_usage": self.total_tokens_used
        }

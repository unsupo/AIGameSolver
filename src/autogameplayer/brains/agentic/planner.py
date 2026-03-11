import random
from autogameplayer.core.models import Observation
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.utils.llm import LLMClientProtocol, extract_json_from_llm_response
from .memory import LongTermMemory

class PlannerAgent:
    """Agent responsible for high-level strategy and goal setting."""
    def __init__(self, client: LLMClientProtocol, model: str, config: GameConfig = None, optimizer=None, knowledge=None, critic=None):
        self.client = client
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.knowledge = knowledge
        self.critic = critic

    async def generate_plan(self, obs: Observation, long_term: LongTermMemory) -> tuple[dict, list[str]]:
        """Synthesizes memory, curiosity, and spatial data into a high-level goal."""
        ctx = obs.state.context
        map_id = ctx.get('map_id', -1)
        pos = (ctx.get('x'), ctx.get('y'))
        
        map_name = self._get_map_name(map_id)
        recalled, recalled_text = await self._query_memory(obs, long_term, map_id, map_name, pos)
        knowledge_context = await self._query_knowledge(map_id, map_name, pos)
        
        dialogue_context = self._get_dialogue_context(obs, ctx)
        stagnation_context = self._get_stagnation_context(map_id, pos, long_term)
        curiosity_context = self._get_curiosity_context(recalled_text, stagnation_context)
        skill_context = self.optimizer.get_skills_for_map(map_id) if self.optimizer else ""
        
        prompt = self._build_prompt(
            obs, map_id, map_name, pos, 
            recalled_text, knowledge_context, 
            dialogue_context, stagnation_context, 
            curiosity_context, skill_context
        )
        
        plan_data = await self._execute_llm_plan(prompt, dialogue_context != "")
        return plan_data, recalled

    def _get_map_name(self, map_id):
        if self.config and self.config.profile and map_id in self.config.profile.maps:
            return self.config.profile.maps[map_id]
        return "Unknown Area"

    async def _query_memory(self, obs, long_term, map_id, map_name, pos):
        query = f"Strategies for Map {map_id} ({map_name}) at position {pos}"
        memories = await long_term.query(query, current_map_id=map_id if isinstance(map_id, int) else None)
        
        loop_context = ""
        for mem in memories:
            if "Loop Detected" in mem and obs.state_hash in mem:
                loop_context = "\n🚨 LOOP AVOIDANCE ACTIVE: Purposefully DEVIATE from previous actions."
                break
        
        recalled_text = "\n".join(f"- {m}" for m in memories) if memories else "No specific knowledge recalled."
        if loop_context:
            recalled_text += loop_context
            
        return memories, recalled_text

    async def _query_knowledge(self, map_id, map_name, pos):
        if not self.knowledge:
            return ""
        snippets = await self.knowledge.query(f"What should I do on Map {map_id} ({map_name}) near {pos}?")
        if snippets:
            return "\nEXTRACTED KNOWLEDGE:\n" + "\n".join(f"- {s}" for s in snippets)
        return ""

    def _get_dialogue_context(self, obs, ctx):
        ocr_upper = (obs.state.ocr_text or "").upper()
        naming_keywords = ["UPPER CASE", "LOWER CASE", "ED IT", "DELETE", " NAME "]
        is_naming = any(k in ocr_upper for k in naming_keywords)
        is_dialogue = ctx.get('is_dialogue', False) or obs.state.has_dialogue_box or obs.state.has_dialogue_arrow
        
        if is_dialogue or is_naming:
            return "\n⚠️ DIALOGUE/UI BUSY: Player CANNOT move. Prioritize 'A', 'B' or 'START'."
        return ""

    def _get_stagnation_context(self, map_id, pos, long_term):
        if not isinstance(map_id, int) or pos[0] is None:
            return ""
        count = long_term.get_visit_count(map_id, pos[0], pos[1])
        if count > 20:
            return f"\n⚠️ SPATIAL STAGNATION ({count} visits): FORCE a new direction."
        return ""

    def _get_curiosity_context(self, loop_text, stagnation_text):
        if "🚨" in loop_text or "⚠️" in stagnation_text:
            return ""
        if random.random() < 0.2:
            return "\n💡 CURIOSITY MODE: Try something unexpected."
        return ""

    def _build_prompt(self, obs, map_id, map_name, pos, recalled_text, knowledge, dialogue, stagnation, curiosity, skills):
        completed = ""
        if self.critic and self.critic.milestones:
            completed = "\nCOMPLETED GLOBAL OBJECTIVES:\n- " + "\n- ".join(list(self.critic.milestones))

        hints = ""
        if self.config and self.config.profile and self.config.profile.memory_map_hints:
            hints = f"\n--- GAME MEMORY MAP HINTS ---\n{self.config.profile.memory_map_hints}\n"

        return f"""
        You are the STRATEGIC PLANNER for an AI playing {self.config.name if self.config else 'Game'}.
        Map: #{map_id} [{map_name}] | Position: {pos}
        OCR Text: "{obs.state.ocr_text}"
        {dialogue} {stagnation} {skills} {knowledge} {curiosity}
        {hints}
        {completed}
        RECALLED EXPERIENCES:
        {recalled_text}

        OUTPUT: JSON ONLY:
        {{
          "goal": "long-term objective",
          "steps": ["Step 1", "Step 2"],
          "abort_condition": "When to abandon",
          "expected_map_after": int or null,
          "high_stakes": bool
        }}
        """

    async def _execute_llm_plan(self, prompt, is_ui_busy):
        try:
            response = await self.client.acreate_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200, temperature=0.7
            )
            plan = extract_json_from_llm_response(response)
            if plan: return plan
        except Exception as e:
            print(f"⚠️ Planning failed: {e}")
        
        if is_ui_busy:
            return {"goal": "Clear UI", "steps": ["Press A"], "abort_condition": "UI Clear", "expected_map_after": None, "high_stakes": False}
        return {"goal": "Explore", "steps": ["Walk"], "abort_condition": "Stagnation", "expected_map_after": None, "high_stakes": False}

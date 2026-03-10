import random
from autogameplayer.core.models import Observation
from autogameplayer.core.config_loader import GameConfig
from autogameplayer.utils.llm import LLMClientProtocol
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
        
        # Get Map Name from config
        map_name = "Unknown Area"
        if self.config and self.config.profile and map_id in self.config.profile.maps:
            map_name = self.config.profile.maps[map_id]
        
        # --- FEATURE: Milestone Injection ---
        # Inject global milestones so the AI knows what it has already accomplished
        completed_milestones = ""
        if self.critic and self.critic.milestones:
            completed_milestones = "\nCOMPLETED GLOBAL OBJECTIVES:\n- " + "\n- ".join(list(self.critic.milestones))
        # ------------------------------------

        # 1. Query RAG for knowledge base insights (README, guides)
        external_context = ""
        recalled = []
        if self.knowledge:
            knowledge_snippets = await self.knowledge.query(f"What should I do on Map {map_id} ({map_name}) near {pos}?")
            recalled.extend(knowledge_snippets)
            if knowledge_snippets:
                external_context = "\nEXTRACTED KNOWLEDGE:\n" + "\n".join(f"- {s}" for s in knowledge_snippets)

        # 2. Query RAG for specific situational failures/rules
        query = f"Strategies for Map {map_id} ({map_name}) at position {pos}"
        specific_memories = await long_term.query(query, current_map_id=map_id if isinstance(map_id, int) else None)
        recalled.extend(specific_memories)
        recalled_text = "\n".join(f"- {m}" for m in specific_memories) if specific_memories else "No specific knowledge recalled."

        # NEW: Check specifically for LOOP WARNINGS for this exact state hash
        loop_context = ""
        for mem in specific_memories:
            if "Loop Detected" in mem and obs.state_hash in mem:
                loop_context = "\n🚨 LOOP AVOIDANCE ACTIVE: You have been in a loop on this exact screen before. PURPOSEFULLY DEVIATE from your previous actions. Try random buttons or a completely different direction."
                break

        # 3. Global Spatial Awareness (World Map)
        exploration_summary = ""
        if isinstance(map_id, int):
            exploration_summary = f"\nWORLD MAP (Map {map_id} [{map_name}]):\\n{long_term.get_explored_summary(map_id)}"

        # NEW: Dialogue Busy Context
        dialogue_busy_context = ""
        ocr_upper = (obs.state.ocr_text or "").upper()
        naming_keywords = ["UPPER CASE", "LOWER CASE", "ED IT", "DELETE", " NAME "]
        is_naming = any(k in ocr_upper for k in naming_keywords)
        if obs.state.has_dialogue_arrow or is_naming:
            dialogue_busy_context = "\n⚠️ DIALOGUE/UI BUSY: The game is waiting for input to advance text or a name. Prioritize 'A' or 'START' actions. Avoid exploration goals until this is cleared."

        # NEW: Heat Map Detection (Spatial SLAM)
        visit_count = 0
        if isinstance(map_id, int) and pos[0] is not None:
            visit_count = long_term.get_visit_count(map_id, pos[0], pos[1])
            
        stagnation_context = ""
        if visit_count > 20:
            stagnation_context = f"\n⚠️ SPATIAL STAGNATION WARNING: You have visited this exact tile {visit_count} times in the past. Your previous strategies in {map_name} are clearly failing to progress. FORCE a completely new direction or random exploration goal. DO NOT repeat your previous sequence."

        # Curiosity Roll: Stochastic experimentation
        curiosity_roll = random.random()
        curiosity_context = loop_context or stagnation_context
        if not curiosity_context and curiosity_roll < 0.2: # 20% chance to "Experiment"
            curiosity_context = """
            💡 CURIOSITY MODE: Ignore the obvious path. Try something unexpected to see if it reveals a new area or reward.
            """

        # 4. Pull relevant skills for this context
        skill_context = ""
        if self.optimizer:
            skill_context = self.optimizer.get_skills_for_map(map_id)

        # 5. Goal Scaling: Dynamically adjust goal complexity based on session length
        goal_scale = "Short-term tactical focus (next 20 steps)."
        if ctx.get('party_count', 0) > 0:
            goal_scale = "Mid-term exploration focus (next 50 steps)."

        prompt = f"""
        You are the STRATEGIC PLANNER for an AI playing {self.config.name}.
        TASK: Synthesize the current state, memory, and completed objectives into a concrete, actionable curriculum.
        
        --- CURRENT CONTEXT ---
        Map: #{map_id} [{map_name}] | Position: {pos}
        Stage: {obs.state.stage.value}
        {completed_milestones}
        OCR Text: "{obs.state.ocr_text}"
        
        WARNINGS/GUIDANCE FROM CRITIC:
        {obs.guidance if obs.guidance else "None"}
        
        {dialogue_busy_context}
        {stagnation_context}
        {skill_context}
        {external_context}
        {curiosity_context}
        {exploration_summary}
        {goal_scale}
        
        RECALLED EXPERIENCES (Area: {map_name}):
        {recalled_text}

        --- POKEMON RED MEMORY MAP HINTS ---
        - 0xD35E = Current Map ID
        - 0xD362 = Player X tile
        - 0xD361 = Player Y tile
        - 0xD16B = Badge count (0-8)
        - 0xD163 = Party size
        - 0xD31C = Player money (3 bytes BCD)

        TASK: Define a high-level strategic goal for the next 50 steps.
        
        CRITICAL: Respond ONLY with a valid JSON object. Do not include markdown formatting, preamble, or any conversational text.

        OUTPUT: JSON ONLY with the following structure:
        {{
          "goal": "A specific, measurable long-term objective (e.g., 'Defeat Brock at Pewter Gym')",
          "steps": ["Step 1", "Step 2", "Step 3"],
          "abort_condition": "When to abandon this specific plan",
          "expected_map_after": int or null,
          "high_stakes": bool
        }}
        """

        try:
            from autogameplayer.utils.llm_utils import extract_json_from_llm_response
            response = await self.client.acreate_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            # Extract JSON from response
            plan_data = extract_json_from_llm_response(response)
            if not plan_data:
                return {
                    "goal": "Explore and progress.",
                    "steps": ["Walk around", "Talk to NPCs"],
                    "abort_condition": "Stagnation",
                    "expected_map_after": None,
                    "high_stakes": False
                }, recalled

            return plan_data, recalled
        except Exception as e:
            print(f"⚠️ Planning failed: {e}")
            return {
                "goal": "Explore and progress.",
                "steps": ["Walk around", "Talk to NPCs"],
                "abort_condition": "Stagnation",
                "expected_map_after": None,
                "high_stakes": False
            }, []

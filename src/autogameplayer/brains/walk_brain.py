import random
from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.registry import Registry


@Registry.register_brain("walk")
class WalkBrain(Brain):
    """A brain that favors walking in a straight line to explore the map."""

    def __init__(self, controller: Controller, **kwargs):
        self.controller = controller
        self.directions = ["up", "down", "left", "right"]
        # Ensure controller has these directions
        self.valid_directions = [
            d for d in self.directions if d in self.controller.buttons
        ]

        if self.valid_directions:
            self.current_direction = random.choice(self.valid_directions)
        else:
            self.current_direction = "a"

        self.steps_in_current_direction = 0
        self.max_steps = random.randint(3, 10)

    async def act(self, observation: Observation) -> Action:

        # 15% chance to press A, B, or START (to get past title screens and menus)
        if random.random() < 0.15:
            # We use fallback_action which picks ANY button, including a, b, start.
            # But we could also just pick from ["a", "b", "start"] specifically.
            # Let's use fallback_action for a generic "do something else" action.
            return self.fallback_action(self.controller)

        if self.steps_in_current_direction >= self.max_steps:
            # Pick a new direction
            self.current_direction = random.choice(self.valid_directions)
            self.steps_in_current_direction = 0
            self.max_steps = random.randint(3, 10)

        self.steps_in_current_direction += 1
        reasoning = f"Walking {self.current_direction.upper()} ({self.steps_in_current_direction}/{self.max_steps})."
        return Action(button=self.current_direction, duration=10, reasoning=reasoning)

from autogameplayer.core.interfaces import Brain, Controller
from autogameplayer.core.models import Observation, Action
from autogameplayer.core.registry import Registry

@Registry.register_brain("random")
class RandomBrain(Brain):
    def __init__(self, controller: Controller, **kwargs):
        self.controller = controller

    async def act(self, observation: Observation) -> Action:
        return self.fallback_action(self.controller)


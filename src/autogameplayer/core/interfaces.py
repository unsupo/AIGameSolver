from abc import ABC, abstractmethod
from typing import List, Optional, Any
from PIL import Image
from autogameplayer.core.models import Observation, Action, StepResult

class IMemoryReader(ABC):
    """Interface for reading and writing to game memory."""
    def read_memory(self, address: int) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} does not support memory reading.")

    def write_memory(self, address: int, value: int):
        """Writes a byte value to a specific memory address."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support memory writing.")

    def read_memory_block(self, address: int, length: int) -> bytes:
        raise NotImplementedError(f"{self.__class__.__name__} does not support memory block reading.")

class IEmulatorCore(ABC):
    """Interface for basic emulator control."""
    @property
    @abstractmethod
    def emulator_name(self) -> str: pass

    @property
    @abstractmethod
    def supported_buttons(self) -> List[str]: pass

    @property
    @abstractmethod
    def has_release(self) -> bool: pass

    @abstractmethod
    def get_screenshot(self) -> Image.Image: pass

    @abstractmethod
    def send_input(self, button: str): pass

    @abstractmethod
    def tick(self, frames: int = 1): pass

    def manage_checkpoint(self, action: str, slot: int):
        raise NotImplementedError(f"{self.__class__.__name__} does not support checkpoints.")

    @abstractmethod
    def close(self): pass

class BaseEmulator(IEmulatorCore, IMemoryReader, ABC):
    """A full-featured emulator backend supporting both control and memory access."""
    pass

class Brain(ABC):
    @abstractmethod
    async def act(self, observation: Observation, mcp_client: Optional[Any] = None) -> Action: pass

    def close(self):
        """Optional cleanup for background tasks."""
        pass

    def fallback_action(self, controller: Any) -> Action:
        """Returns a random action as a fallback."""
        import random
        btns = getattr(controller, "buttons", ["a", "up"])
        return Action(button=random.choice(btns), duration=5)

class Environment(ABC):
    @abstractmethod
    async def reset(self) -> Observation: pass
    @abstractmethod
    async def step(self, action: Action) -> StepResult: pass

class RewardFunction(ABC):
    @abstractmethod
    async def compute(self, prev_obs: Observation, obs: Observation) -> float: pass

class Controller(ABC):
    @property
    @abstractmethod
    def buttons(self) -> List[str]: pass

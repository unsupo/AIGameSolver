from typing import Dict, Type
from autogameplayer.core.interfaces import Brain, RewardFunction, Controller, BaseEmulator

class Registry:
    _BRAINS: Dict[str, Type[Brain]] = {}
    _REWARDS: Dict[str, Type[RewardFunction]] = {}
    _CONTROLLERS: Dict[str, Type[Controller]] = {}
    _EMULATORS: Dict[str, Type[BaseEmulator]] = {}

    @classmethod
    def register_brain(cls, name: str):
        def decorator(subclass):
            cls._BRAINS[name] = subclass
            return subclass
        return decorator

    @classmethod
    def register_reward(cls, name: str):
        def decorator(subclass):
            cls._REWARDS[name] = subclass
            return subclass
        return decorator

    @classmethod
    def register_controller(cls, name: str):
        def decorator(subclass):
            cls._CONTROLLERS[name] = subclass
            return subclass
        return decorator

    @classmethod
    def register_emulator(cls, extensions: list):
        def decorator(subclass):
            for ext in extensions:
                cls._EMULATORS[ext.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def create_brain(cls, name: str, controller: Controller, **kwargs) -> Brain:
        if name not in cls._BRAINS:
            raise ValueError(f"Brain '{name}' not found in registry. Available: {list(cls._BRAINS.keys())}")
        return cls._BRAINS[name](controller, **kwargs)

    @classmethod
    def create_reward(cls, name: str, **kwargs) -> RewardFunction:
        if name not in cls._REWARDS:
            raise ValueError(f"Reward '{name}' not found in registry. Available: {list(cls._REWARDS.keys())}")
        return cls._REWARDS[name](**kwargs)

    @classmethod
    def create_controller(cls, name: str) -> Controller:
        if name not in cls._CONTROLLERS:
            # Fallback to standard if not found
            return cls._CONTROLLERS.get("standard", cls._CONTROLLERS.get("gb"))()
        return cls._CONTROLLERS[name]()

    @classmethod
    def create_emulator(cls, rom_path: str) -> BaseEmulator:
        import os
        ext = os.path.splitext(rom_path)[1].lower()
        if ext not in cls._EMULATORS:
            # Fallback to common extensions if unknown
            fallback_ext = ".gb" if ".gb" in cls._EMULATORS else ".gba"
            if fallback_ext in cls._EMULATORS:
                print(f"⚠️ Unknown extension '{ext}'. Falling back to {fallback_ext} emulator.")
                return cls._EMULATORS[fallback_ext](rom_path)
            
            raise ValueError(f"No emulator found for extension '{ext}'. Available: {list(cls._EMULATORS.keys())}")
        return cls._EMULATORS[ext](rom_path)

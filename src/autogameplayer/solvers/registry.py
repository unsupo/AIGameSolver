from typing import Type, Dict
import inspect
from autogameplayer.core.solver import BaseSolver


class SolverRegistry:
    """
    A central registry for all solver implementations to simplify modular plugging
    and ensemble orchestration.
    """

    _SOLVERS: Dict[str, Type[BaseSolver]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a BaseSolver class.
        e.g. @SolverRegistry.register("random")
        """

        def decorator(solver_cls: Type[BaseSolver]):
            if not inspect.isclass(solver_cls) or not issubclass(
                solver_cls, BaseSolver
            ):
                raise TypeError(f"{solver_cls} must be a subclass of BaseSolver.")
            cls._SOLVERS[name] = solver_cls
            return solver_cls

        return decorator

    @classmethod
    def get_solver(cls, name: str, **kwargs) -> BaseSolver:
        if name not in cls._SOLVERS:
            raise ValueError(f"Solver '{name}' not found in registry.")
        return cls._SOLVERS[name](**kwargs)

    @classmethod
    def list_solvers(cls) -> Dict[str, Type[BaseSolver]]:
        return cls._SOLVERS.copy()

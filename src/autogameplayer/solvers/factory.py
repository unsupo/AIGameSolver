from autogameplayer.core.config_loader import SolverConfig
from autogameplayer.core.solver import BaseSolver
from autogameplayer.solvers.registry import SolverRegistry


class SolverFactory:
    """
    Recursively instantiates solvers and decorators from a SolverConfig tree.
    """

    @classmethod
    def create_solver(cls, config: SolverConfig, **kwargs) -> BaseSolver:
        solver_type = config.type
        params = config.params.copy()
        params.update(kwargs)

        # 1. Handle Recursive Solver Trees (Ensembles/Routers)
        if config.solvers:
            sub_solvers = {}
            for name, sub_config in config.solvers.items():
                sub_solvers[name] = cls.create_solver(sub_config, **kwargs)
            params["solvers"] = sub_solvers

        # 2. Handle Decorators (base_solver)
        if config.base_solver:
            params["base_solver"] = cls.create_solver(config.base_solver, **kwargs)

        # 3. Handle Fallback Solvers (routers)
        if config.fallback_solver:
            params["fallback_solver"] = cls.create_solver(
                config.fallback_solver, **kwargs
            )
        if "default_solver" in params and isinstance(params["default_solver"], SolverConfig):
             params["default_solver"] = cls.create_solver(params["default_solver"], **kwargs)

        # 4. Instantiate from Registry
        return SolverRegistry.get_solver(solver_type, **params)

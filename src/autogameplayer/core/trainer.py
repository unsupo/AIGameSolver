import json
import os
import asyncio
import logging
import shutil
from autogameplayer.core.config_loader import load_game_config, GameConfig
from autogameplayer.core.curriculum import Curriculum
from autogameplayer.core.population import PopulationManager
from autogameplayer.orchestrator.orchestrator import Orchestrator
from autogameplayer.core.config import settings
from autogameplayer.core.registry import Registry
from autogameplayer.utils.launcher import ServerLauncher
from autogameplayer.utils.process import get_base_env
from autogameplayer.core.mcp_client import MCPClient

class GeneticTrainer:
    """High-level Training Orchestrator with Curriculum Learning."""
    def __init__(
        self, 
        config: GameConfig,
        population_size: int = 20, 
        num_workers: int = 4
    ):
        self.config = config
        self.rom_path = config.rom
        
        # Determine genome shape from the specified controller
        controller = Registry.create_controller(config.controller)
        num_buttons = len(controller.buttons)
        if num_buttons == 0:
            # Fallback if dynamic controller hasn't synced yet
            num_buttons = 10 if self.rom_path.lower().endswith(".gba") else 8
            
        genome_shape = (384, num_buttons)
        
        self.pop_manager = PopulationManager(population_size, genome_shape)
        self.orchestrator = Orchestrator(num_workers, self.rom_path)
        self.metrics_path = settings.base_dir / "logs" / "metrics" / "evolution_stats.json"
        self.history = []
        
        # Curriculum state from profile
        checkpoints = config.profile.curriculum if config.profile else []
        self.curriculum = Curriculum(checkpoints)
        self.success_threshold = 0.2 # 20% of population must reach the goal to advance

    async def _check_initial_checkpoint(self):
        """Checks if the starting state already meets the first few curriculum goals."""
        if not self.curriculum.checkpoints:
            return

        print("🔍 Checking initial state against curriculum...")
        port = 8998
        launcher = ServerLauncher(self.rom_path, port)
        launcher.start(env_vars=get_base_env())
        
        client = MCPClient(f"http://localhost:{port}/sse")
        try:
            if not await launcher.wait_until_healthy():
                return
                
            await client.connect()
            
            # Load initial state
            try:
                await client.call_tool("manage_checkpoint", {"action": "load", "slot": 0})
            except Exception as e:
                logging.debug(f"No initial state found in slot 0: {e}")
            
            # Check milestones
            for target in self.curriculum.checkpoints:
                val = await client.call_tool("read_memory", {"address": target.address})
                if int(val) >= target.target_value:
                    print(f"✅ Initial state already completed '{target.name}'.")
                    self.curriculum.advance()
                else:
                    break
        finally:
            try: await client.disconnect()
            except Exception: pass
            launcher.stop()

    def train(self, generations: int = 50, steps_per_episode: int = 200):
        print("🚀 Starting Curriculum Evolution...")
        os.makedirs(settings.base_dir / "models", exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        
        if not self.curriculum.checkpoints:
            print("⚠️ Warning: No curriculum defined in profile. Training might be aimless.")

        # Check initial progress
        asyncio.run(self._check_initial_checkpoint())
        
        for gen in range(1, generations + 1):
            if self.curriculum.is_complete():
                print("🏁 ALL CHECKPOINTS COMPLETE!")
                break
                
            target = self.curriculum.get_active_target()
            target_name = target.name if target else "General Exploration"
            target_desc = target.description if target else "No specific goal"
            
            print(f"\n--- Gen {gen} | Target: {target_name} ({target_desc}) ---")
            
            genomes = self.pop_manager.get_population()
            results = self.orchestrator.evaluate_population(
                genomes, 
                steps_per_episode=steps_per_episode,
                target_checkpoint=target
            )
            
            fitness_scores = [r["reward"] for r in results]
            success_count = sum(1 for r in results if r["reached_target"])
            success_rate = success_count / len(genomes)
            
            # Stats
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            max_fitness = max(fitness_scores)
            print(f"Summary: Success Rate: {success_rate:.1%}, Max Reward: {max_fitness:.2f}")
            
            # Save metrics
            self.history.append({
                "generation": gen,
                "target": target_name,
                "success_rate": success_rate,
                "max_fitness": float(max_fitness),
                "avg_fitness": float(avg_fitness)
            })
            with open(self.metrics_path, "w") as f: json.dump(self.history, f)

            # ADVANCE CURRICULUM if threshold met
            if target and success_rate >= self.success_threshold:
                print(f"✨ MILESTONE REACHED! {success_count} individuals achieved '{target.name}'.")
                # Move the 'winning' state from slot 99 to slot 0 (global start)
                state_99 = settings.base_dir / "state_slot_99.state"
                state_0 = settings.base_dir / "state_slot_0.state"
                if os.path.exists(state_99):
                    shutil.copy(state_99, state_0)
                    print("💾 Advancing starting line to the new checkpoint.")
                
                self.curriculum.advance()
                if self.curriculum.is_complete():
                    print("🏁 ALL CHECKPOINTS COMPLETE! Game finished?")
                    break
            
            self.pop_manager.evolve(fitness_scores)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evolutionary Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to game config YAML")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    config = load_game_config(args.config)
    
    trainer = GeneticTrainer(
        config=config,
        population_size=args.pop_size,
        num_workers=args.workers
    )
    trainer.train(generations=args.generations, steps_per_episode=args.steps)

if __name__ == "__main__":
    main()

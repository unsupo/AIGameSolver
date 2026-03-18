import concurrent.futures
import numpy as np
import os
import signal
from typing import List
from autogameplayer.core.config import settings
from autogameplayer.vision.encoder import VisionEncoder
from autogameplayer.worker.worker import run_worker_episode
from autogameplayer.core.config_loader import discover_roms, GameConfig


class Orchestrator:
    """Manages mass parallelism across workers."""

    def __init__(self, num_workers: int, rom_path: str, config: GameConfig = None):
        self.num_workers = num_workers
        self.rom_path = rom_path
        self.config = config

    def evaluate_population(
        self,
        genomes: List[np.ndarray],
        steps_per_episode: int = 100,
        target_checkpoint=None,
        initial_slot: int = 0,
    ) -> List[dict]:
        population_size = len(genomes)
        print(
            f"Starting evaluation of {population_size} individuals using {self.num_workers} workers..."
        )
        results = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            try:
                futures = []
                for i, genome in enumerate(genomes):
                    # Pass initial_slot and config to each worker
                    futures.append(
                        executor.submit(
                            run_worker_episode,
                            i,
                            self.rom_path,
                            steps_per_episode,
                            genome,
                            target_checkpoint,
                            initial_slot,
                            self.config,
                        )
                    )
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        res = future.result()
                        results.append(res)
                        print(
                            f"Individual {i + 1}/{population_size} evaluated. Reward: {res['reward']:.2f}, Goal Reached: {res['reached_target']}"
                        )
                    except Exception as e:
                        print(f"⚠️ Error evaluating individual: {e}")
                        results.append({"reward": -100.0, "reached_target": False})
            except KeyboardInterrupt:
                print("\n🛑 Training interrupted! Shutting down pool...")
                executor.shutdown(wait=False, cancel_futures=True)
                # Kill worker processes directly if possible (Unix)
                if os.name != "nt":
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                raise

        return results


def main():
    # Simple CLI entry point for the orchestrator
    roms = discover_roms()
    if not roms:
        print("❌ Error: No ROM found.")
        return

    rom_path = str(roms[0])
    orchestrator = Orchestrator(num_workers=4, rom_path=rom_path)

    # Evaluate a population of 8 using 4 parallel workers
    vision_dim = VisionEncoder.get_dim(settings.vision_model)
    dummy_genomes = [np.random.randn(vision_dim * 8) for _ in range(8)]
    results = orchestrator.evaluate_population(dummy_genomes, steps_per_episode=50)
    print("Final Population Results:", results)


if __name__ == "__main__":
    main()

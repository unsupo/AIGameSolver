import numpy as np
from typing import List, Tuple

class PopulationManager:
    """Manages a population of genomes for evolutionary scaling."""
    def __init__(self, population_size: int, genome_shape: Tuple[int, int]):
        self.population_size = population_size
        self.genome_shape = genome_shape
        self.genome_size = np.prod(genome_shape)
        # Initialize random population
        self.population = [np.random.randn(self.genome_size) for _ in range(population_size)]

    def get_population(self) -> List[np.ndarray]:
        return self.population

    def evolve(self, fitness_scores: List[float], mutation_rate: float = 0.1):
        """Creates the next generation based on fitness."""
        # 1. Selection (Top 25% elitism)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        num_elites = max(1, self.population_size // 4)
        elites = [self.population[i] for i in sorted_indices[:num_elites]]
        
        new_population = list(elites)
        
        # 2. Reproduction & Mutation
        while len(new_population) < self.population_size:
            # Pick a parent from elites
            parent = elites[np.random.randint(len(elites))]
            
            # Simple mutation
            child = parent + np.random.randn(self.genome_size) * mutation_rate
            new_population.append(child)
            
        self.population = new_population
        return self.population

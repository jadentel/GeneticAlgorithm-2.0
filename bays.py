import random
from bayes_opt import BayesianOptimization
import time

from Conformation import Conformation, Protein
from Population import Population

# Global variables as before.
global_fittest_ptr = None
isTerminated = False
switch_enable_graphics = False  
switch_minen = -22              
switch_max_evaluations = 50000  

def calculation(pop: Population):
    global global_fittest_ptr, isTerminated
    global_fittest_ptr = pop.getFittest()
    
    # Continue until the fittest's fitness reaches threshold or max evaluations are exceeded.
    while pop.getFittest().getFitness() > switch_minen and Conformation.energyEvalSteps < switch_max_evaluations:
        pop.crossover()
        
        if pop.getFittest().getFitness() < global_fittest_ptr.getFitness():
            global_fittest_ptr = pop.getFittest()
            if not switch_enable_graphics:
                print(global_fittest_ptr.getStatusString())
                global_fittest_ptr.printAsciiPicture()
    
    isTerminated = True

def run_ga(population_size: float, mutation_probability: float, crossover_probability: float) -> float:
    """
    Run the genetic algorithm with given hyperparameters and return the negative final fitness.
    We return the negative fitness because the GA minimizes fitness and BayesianOptimization maximizes.
    """
    # Convert population_size to integer for the GA.
    pop_size = int(population_size)
    # Use the same protein sequence as before.
    prot = Protein("BBWBBWWBBWWBBBBBWWWWWWWWWWBBBBBBWWBBWWBBWBBWWWWW")
    
    # Reset evaluation counter for fairness.
    Conformation.energyEvalSteps = 0
    # Create the Population instance with the given hyperparameters.
    pop = Population(pop_size, prot, mutation_probability, crossover_probability)
    
    # Run the GA.
    calculation(pop)
    
    fittest = pop.getFittest()
    final_fitness = fittest.getFitness()
    print(f"Run complete with params: pop_size={pop_size}, mut_prob={mutation_probability}, cross_prob={crossover_probability} => final fitness: {final_fitness}")
    
    # Return negative fitness so that a lower fitness (better) gives a higher objective.
    return -final_fitness

def bayesian_optimization():
    # Define the parameter bounds.
    pbounds = {
        'population_size': (500, 1000),
        'mutation_probability': (0.01, 0.2),
        'crossover_probability': (0.7, 0.9)
    }
    
    optimizer = BayesianOptimization(
        f=run_ga,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    # Run with a few initial random points and then iterations.
    optimizer.maximize(init_points=5, n_iter=10)
    return optimizer.max

def compare_default_vs_bayesian():
    # Default hyperparameters from your original code.
    default_params = {
        'population_size': 600,
        'mutation_probability': 0.02,
        'crossover_probability': 0.8
    }
    print("\n--- Running GA with Default Parameters ---")
    default_score = run_ga(**default_params)
    print("Default GA negative fitness:", default_score)
    
    print("\n--- Running Bayesian Optimization for Hyperparameter Tuning ---")
    best = bayesian_optimization()
    best_params = best['params']
    best_score = best['target']
    print("\nBest hyperparameters found:", best_params)
    print("Best GA negative fitness with optimized parameters:", best_score)
    
    # Compare results.
    if best_score > default_score:
        print("\nBayesian optimized hyperparameters performed better!")
    else:
        print("\nDefault parameters performed better!")
    
def main():
    # Uncomment one of the options below:
    
    # Option 1: Run the original GA
    # prot = Protein("BBBBWWWWWBBBBBBBBBBBBWWWWWWBBBBBBBBBBBWWWWBBBBBBBBBBBBWWWBBBBBBBBBBBBWWWWWWBBWBBWWBW")
    # pop = Population(600, prot, 0.02, 0.8)
    # calculation(pop)
    # fittest = pop.getFittest()
    # print("Final Fittest Individual:", fittest.getStatusString())
    # fittest.printAsciiPicture()
    
    # Option 2: Run the comparison between default and Bayesian-tuned parameters.
    compare_default_vs_bayesian()
    
if __name__ == "__main__":
    main()

import random

from Conformation import Conformation, Protein
from Population import Population

# Global variables
global_fittest_ptr = None
isTerminated = False
switch_enable_graphics = False  # Set to True to enable graphics
switch_minen = -22              # Example threshold for fitness
switch_max_evaluations = 500000  # Maximum number of energy evaluations

def calculation(pop: Population):
    global global_fittest_ptr, isTerminated
    global_fittest_ptr = pop.getFittest()
    
    # Continue until the fittest's fitness reaches threshold or max evaluations are exceeded.
    while pop.getFittest().getFitness() > switch_minen and Conformation.energyEvalSteps < switch_max_evaluations:
        pop.crossover()
        
        # Check if an improvement has been made.
        if pop.getFittest().getFitness() < global_fittest_ptr.getFitness():
            global_fittest_ptr = pop.getFittest()
            
            # If graphics is disabled: output ASCII status and picture to console
            if not switch_enable_graphics:
                print(global_fittest_ptr.getStatusString())
                global_fittest_ptr.printAsciiPicture()
    
    isTerminated = True

def main():
    # Seed the random number generator for reproducibility
    #random.seed(42)
    
    # Initialize a Protein object with an example sequence
    
    prot = Protein("BBWBBWWBBWWBBBBBWWWWWWWWWWBBBBBBWWBBWWBBWBBWWWWW")
                  # 20 = BWBWWBBWBWWBWBBWWBWB = -9
                  # 24 = BBWWBWWBWWBWWBWWBWWBWWBB = -9
                  # 25 = BBWBBWWBBBBWWBBBBWWBBBBWW = -8
                  # 36 = WWWBBWWBBWWWWWBBBBBBBWWBBWWWWBBWWBWW = -14
                  # 48 = WWBWWBBWWBBWWWWWBBBBBBBBBBWWWWWWBBWWBBWWBWWBBBBB = -23
                  # 50 = BBWBWBWBWBBBBWBWWWBWWWBWWWWBWWWBWWWBWBBBBWBWBWBWBB = -21
                  # 60 = WWBBBWBBBBBBBBWWWBBBBBBBBBBWBWWWBBBBBBBBBBBBWWWWBBBBBBWBBWBW = -34
                  # 64 = BBBBBBBBBBBBWBWBWWBBWWBBWWBWWBBWWBBWWBWWBBWWBBWWBWBWBBBBBBBBBBBB = -42
                  # 85 = BBBBWWWWWBBBBBBBBBBBBWWWWWWBBBBBBBBBBBWWWWBBBBBBBBBBBBWWWBBBBBBBBBBBBWWWWWWBBWBBWWBW = -52

      
    population_size = 952
    mutation_probability = 0.05373444298553897
    crossover_probability = 0.705457530130113

    
    # Create the Population object
    pop = Population(population_size, prot, mutation_probability, crossover_probability)
    
    # Run the calculation loop
    calculation(pop)
    
    # Output the final fittest individual
    fittest = pop.getFittest()
    print("Final Fittest Individual:", fittest.getStatusString())
    fittest.printAsciiPicture()

if __name__ == "__main__":
    main()




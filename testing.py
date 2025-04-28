
"""
Silent GA Benchmark Script for 24-Residue Protein Folding
- Suppresses verbose output during Population initialization
- Runs the GA multiple times
- Logs BestEnergy, EvalToBest, UniqueConformations to CSV
- Analyzes results with summary statistics and a simple test
"""
import csv
import statistics
import io
import contextlib
from Conformation import Conformation, Protein
from Population import Population

# Configuration constants
SEQUENCE = "BBBBBBBBBBBBWBWBWWBBWWBBWWBWWBBWWBBWWBWWBBWWBBWWBWBWBBBBBBBBBBBB"
POPULATION_SIZE = 1000
MUTATION_PROBABILITY = 0.05
CROSSOVER_PROBABILITY = 0.85
SWITCH_MIN_ENERGY = -42
SWITCH_MAX_EVALUATIONS = 100000
RUNS = 5
CSV_FILENAME = "ga_24seq_results.csv"
OPTIMAL_ENERGY = -42


def create_silent_population(size, prot, mut_prob, cross_prob):
    """
    Create a Population without printing its verbose progress.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop = Population(size, prot, mut_prob, cross_prob)
    return pop


def calculation(pop: Population):
    """
    Run GA until threshold or max evals. Return best fitness, evals to best, unique conf count, generations to best, birth generation.
    """
    Conformation.energyEvalSteps = 0
    best = pop.getFittest()
    best_fitness = best.getFitness()
    best_eval = Conformation.energyEvalSteps
    generation = 0
    best_generation = 0
    birth_generation = best.getGeneration()

    while best.getFitness() > SWITCH_MIN_ENERGY and Conformation.energyEvalSteps < SWITCH_MAX_EVALUATIONS:
        pop.crossover()
        generation += 1
        current = pop.getFittest()
        if current.getFitness() < best_fitness:
            best_fitness = current.getFitness()
            best_eval = Conformation.energyEvalSteps
            best_generation = generation
            birth_generation = current.getGeneration()
            best = current
    unique_confs = len(pop.setOfConformations)
    return best_fitness, best_eval, unique_confs, best_generation, birth_generation



def run_multiple_and_log():
    """
    Executes the GA RUNS times, logs metrics to CSV, and prints each run's result.
    """
    prot = Protein(SEQUENCE)
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Run", "BestEnergy", "EvalToBest", "UniqueConformations", "GenerationsToBest", "BirthGenerationOfBest"])
        for i in range(1, RUNS + 1):
            pop = create_silent_population(
                POPULATION_SIZE, prot, MUTATION_PROBABILITY, CROSSOVER_PROBABILITY
            )
            best_energy, evals_to_best, unique_confs, generations_to_best, birth_generation = calculation(pop)
            writer.writerow([i, best_energy, evals_to_best, unique_confs, generations_to_best, birth_generation])
            print(f"Run {i}: BestEnergy={best_energy}, EvalToBest={evals_to_best}, UniqueConfs={unique_confs}, GenerationsToBest={generations_to_best}, BirthGenerationOfBest={birth_generation}")


def analyze_results():
    """
    Reads the CSV, checks runs, and computes mean/std of evaluations, energies, generations, and birth generations.
    """
    energies = []
    evals = []
    generations = []
    births = []
    success_count = 0
    with open(CSV_FILENAME, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            energy = int(row['BestEnergy'])
            eval = int(row['EvalToBest'])
            gen = int(row['GenerationsToBest'])
            birth = int(row['BirthGenerationOfBest'])
            energies.append(energy)
            evals.append(eval)
            generations.append(gen)
            births.append(birth)
            if energy <= OPTIMAL_ENERGY:
                success_count += 1

    mean_energy = statistics.mean(energies)
    stdev_energy = statistics.stdev(energies)
    mean_eval = statistics.mean(evals)
    stdev_eval = statistics.stdev(evals)
    mean_gen = statistics.mean(generations)
    stdev_gen = statistics.stdev(generations)
    mean_birth = statistics.mean(births)
    stdev_birth = statistics.stdev(births)

    min_eval = min(evals)
    max_eval = max(evals)

    print("\n=== GA Performance Summary ===")
    print(f"Protein: {SEQUENCE}")
    print(f"Length: {Protein(SEQUENCE).getLength()}")
    print(f"Optimal target energy: {OPTIMAL_ENERGY}")
    print(f"Runs executed: {RUNS}")
    print(f"Runs reaching optimal energy: {success_count}/{RUNS}")
    print(f"Mean best energy: {mean_energy:.2f}")
    print(f"Std deviation of best energy: {stdev_energy:.2f}")
    print(f"Mean evaluations to best: {mean_eval:.2f}")
    print(f"Std deviation of evaluations: {stdev_eval:.2f}")
    print(f"Mean generations to best: {mean_gen:.2f}")
    print(f"Std deviation of generations: {stdev_gen:.2f}")
    print(f"Mean birth generation of best: {mean_birth:.2f}")
    print(f"Std deviation of birth generations: {stdev_birth:.2f}")
    print(f"Min evaluations: {min_eval}")
    print(f"Max evaluations: {max_eval}")



if __name__ == "__main__":
    print(f"Starting {RUNS} silent GA runs on sequence of length {len(SEQUENCE)}...\n")
    run_multiple_and_log()
    analyze_results()

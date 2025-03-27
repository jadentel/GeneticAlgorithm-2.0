import random
import math
from typing import Optional, Set, List, Tuple

# Direction constants: in the original C++ code the directions were generated as -1, 0, 1.
# We map: -1 -> LEFT, 0 -> FORWARD, 1 -> RIGHT.
FORWARD = 0
LEFT = -1
RIGHT = 1

class Protein:
    def __init__(self, sequence: str):
        self.sequence = sequence

    def getNth(self, i: int) -> str:
        return self.sequence[i]

    def getLength(self) -> int:
        return len(self.sequence)

class Conformation:
    # Static counter
    energyEvalSteps = 0

    def __init__(self, protein: Optional[Protein] = None, setOfPoints: Optional[Set[Tuple[int, int]]] = None):
        # default values (mimicking the default constructor)
        self.protein: Optional[Protein] = protein
        self.setOfPoints: Optional[Set[Tuple[int, int]]] = setOfPoints
        self.length: int = protein.getLength() if protein else 0
        # encoding holds directions for positions 2..n, so length-2 entries
        self.encoding: List[int] = [0] * (self.length - 2) if self.length >= 2 else []
        # absolute positions for each amino acid, as (x,y) tuples
        self.absPositions: List[Tuple[int, int]] = [(0, 0)] * self.length
        self.fitness: int = 0
        self.generation: int = 0
        self.validState: bool = False

        if protein is not None and setOfPoints is not None:
            self.generateRandomConformation(valid=True)

    @classmethod
    def crossover(cls, p1: "Conformation", p2: "Conformation",
                  setOfPoints: Optional[Set[Tuple[int, int]]] = None) -> "Conformation":
        """Recombination (crossover) constructor."""
        new_conf = cls()
        new_conf.setOfPoints = setOfPoints
        new_conf.protein = p1.protein
        new_conf.length = p1.length
        new_conf.generation = (p1.generation + p2.generation) // 2 + 1
        new_conf.encoding = [0] * (new_conf.length - 2)
        new_conf.absPositions = [(0, 0)] * new_conf.length

        # choose random crossover point in range [0, length-2)
        if new_conf.length - 2 > 0:
            randI = random.randint(0, new_conf.length - 3)
        else:
            randI = 0

        # first part from p1, second from p2
        for i in range(0, randI):
            new_conf.encoding[i] = p1.encoding[i]
        for i in range(randI, new_conf.length - 2):
            new_conf.encoding[i] = p2.encoding[i]

        if new_conf.setOfPoints is not None:
            new_conf.calcValidity()
        return new_conf

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Conformation):
            return NotImplemented
        if self.length != other.length:
            return False
        for i in range(self.length - 2):
            if self.encoding[i] != other.encoding[i]:
                return False
        return True

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def getEncoding(self) -> List[int]:
        return self.encoding

    def getProtein(self) -> Protein:
        return self.protein

    def isValid(self) -> bool:
        return self.validState

    def getLength(self) -> int:
        return self.length

    def calcFitness(self):
        Conformation.energyEvalSteps += 1
        fitness = 0
        # For each amino acid (using positions 0..length-1)
        for i in range(self.length):
            # Check if it is hydrophobic ("B")
            if self.protein.getNth(i) == 'B':
                ori = self.absPositions[i]
                # For every amino acid after i+1 (skip the immediate successor)
                for j in range(i + 2, self.length):
                    if self.protein.getNth(j) == 'B':
                        dest = self.absPositions[j]
                        distX = abs(ori[0] - dest[0])
                        distY = abs(ori[1] - dest[1])
                        if (distX == 1 and distY == 0) or (distX == 0 and distY == 1):
                            fitness -= 1
        self.fitness = fitness

    def getFitness(self) -> int:
        return self.fitness

    def getConformationString(self) -> str:
        result = ""
        for d in self.encoding:
            if d == FORWARD:
                result += "F"
            elif d == LEFT:
                result += "L"
            elif d == RIGHT:
                result += "R"
            else:
                result += "?"
        return result

    def generateRandomConformation(self, valid: bool = False):
        # Initialize encoding with random directions in {-1,0,1}
        for i in range(self.length - 2):
            self.encoding[i] = random.randint(-1, 1)
        if valid:
            # repeatedly mutate until a valid conformation is found
            self.calcValidity()
            while not self.validState:
                self.mutate(0.1)
                self.calcValidity()

    def calcValidity(self):
        self.calcAbsolutePositions()
        # Use a temporary set to check for duplicate positions.
        seen: Set[Tuple[int, int]] = set()
        self.validState = True
        for pos in self.absPositions:
            if pos in seen:
                self.validState = False
                return
            seen.add(pos)
        if self.setOfPoints is not None:
            # update external set if provided
            self.setOfPoints.clear()
            self.setOfPoints.update(seen)

    def mutate(self, probability: float):
        for i in range(self.length - 2):
            if self.randomFloat() <= probability:
                # assign a new random direction
                self.encoding[i] = random.randint(-1, 1)

    def calcAbsolutePositions(self):
        # Starting positions:
        # pos0 = (0, 0)
        # pos1 = (0, 1)
        self.absPositions = [None] * self.length
        x, y = 0, 0
        self.absPositions[0] = (x, y)
        y = 1
        self.absPositions[1] = (x, y)

        # lastDirection: 0 means up, 1 means down, 2 means right, 3 means left.
        lastDirection = 0  # starting "up" direction
        for i in range(self.length - 2):
            d = self.encoding[i]
            if lastDirection == 0:  # up
                if d == FORWARD:
                    y += 1
                elif d == RIGHT:
                    x += 1
                    lastDirection = 2
                elif d == LEFT:
                    x -= 1
                    lastDirection = 3
            elif lastDirection == 1:  # down
                if d == FORWARD:
                    y -= 1
                elif d == RIGHT:
                    x -= 1
                    lastDirection = 3
                elif d == LEFT:
                    x += 1
                    lastDirection = 2
            elif lastDirection == 2:  # right
                if d == FORWARD:
                    x += 1
                elif d == RIGHT:
                    y -= 1
                    lastDirection = 1
                elif d == LEFT:
                    y += 1
                    lastDirection = 0
            elif lastDirection == 3:  # left
                if d == FORWARD:
                    x -= 1
                elif d == RIGHT:
                    y += 1
                    lastDirection = 0
                elif d == LEFT:
                    y -= 1
                    lastDirection = 1
            self.absPositions[i + 2] = (x, y)

    def getGeneration(self) -> int:
        return self.generation

    def olden(self):
        self.generation += 1

    def getStatusString(self) -> str:
        return f"Fitness: {self.fitness}   Generation: {self.generation}"

    def printAsciiPicture(self):
        # Determine boundaries
        xs = [pos[0] for pos in self.absPositions]
        ys = [pos[1] for pos in self.absPositions]
        lowestX = min(xs)
        highestX = max(xs)
        lowestY = min(ys)
        highestY = max(ys)
        # Stretch boundaries by factor 2 (as in original)
        width = (highestX - lowestX) * 2 + 1
        height = (highestY - lowestY) * 2 + 1

        # Create a 2D grid of spaces
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Normalize and plot points
        # We'll track the previous point to draw connecting lines.
        lastX, lastY = self.absPositions[0]
        for idx, (x, y) in enumerate(self.absPositions):
            normX = (x - lowestX) * 2
            normY = (y - lowestY) * 2
            # Place the amino acid symbol.
            # Map: 'B' -> '#' and 'W' -> 'O'
            acid = self.protein.getNth(idx)
            if acid == 'B':
                grid[normY][normX] = 'H'
            elif acid == 'W':
                grid[normY][normX] = 'P'
            else:
                grid[normY][normX] = acid

            if idx > 0:
                # Draw connecting line between last and current
                if lastX > x:
                    if normX + 1 < width:
                        grid[normY][normX + 1] = '-'
                elif lastX < x:
                    if normX - 1 >= 0:
                        grid[normY][normX - 1] = '-'
                if lastY < y:
                    if normY - 1 >= 0:
                        grid[normY - 1][normX] = '|'
                elif lastY > y:
                    if normY + 1 < height:
                        grid[normY + 1][normX] = '|'
            lastX, lastY = x, y

        # Print the grid row by row
        for row in grid:
            print(''.join(row))

    def randomFloat(self) -> float:
        # Returns a random float in [0,1)
        return random.random()

    def getAbsAt(self, i: int) -> Tuple[int, int]:
        return self.absPositions[i]

"""# Example usage (similar to the commented-out main in C++):

if __name__ == '__main__':
    # Create a Protein object (e.g., "BWB")
    protein = Protein("BWBWWBBWBWWBWBBWWBWB")
    # Create a set for points
    sop: Set[Tuple[int, int]] = set()

    # Create two Conformation objects
    conf1 = Conformation(protein, sop)
    conf2 = Conformation(protein, sop)

    print("Encoding 1:", conf1.getConformationString())
    print("Encoding 2:", conf2.getConformationString())

    # Create a new Conformation using crossover
    confCrossover = Conformation.crossover(conf1, conf2, sop)
    print("Encoding after Crossover:", ''.join(
        "F" if d == FORWARD else "L" if d == LEFT else "R" if d == RIGHT else "?"
        for d in confCrossover.getEncoding()))
    
    if conf1 != conf2:
        print("Conformations are different!")
    else:
        print("Conformations are the same!")

    # Optionally, display the ASCII picture of a conformation
    print("\nASCII picture of conf1:")
    conf1.printAsciiPicture()
"""
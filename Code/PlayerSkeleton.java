import java.util.Arrays;

// class Individual contains methods related to individuals within a population
class Individual implements Comparable<Individual>{
	 int MUTATION_RATE=0; // (0,1) value that describes the chance with which a mutation occurs
	 int NUM_WEIGHTS = 6; // number of heuristics used
	 double[] weights; // weight for each heuristic
	 int gameScore; // result after playing games; equivalent to lines cleared after game

	// the constructor initializes the weights randomly. All weights are within range (-10, 0) except for
	// linesCleared, which is in the range (0, 10)
	public Individual(){
		weights = new double[NUM_WEIGHTS];
		for (int i = 0; i<NUM_WEIGHTS; i++){
			weights[i] = Math.random()*(-10);
		}
		weights[1] = -weights[1]; //weights[1] represents linesCleared

	}

	// cross two individuals and returns the 2 children
	public Individual[] cross(Individual p2) {
		Individual[] children = new Individual[2];
		for (int i = 0; i < NUM_WEIGHTS; i++) {
			if (Math.random() > 0.5) {
				children[0].weights[i] = this.weights[i];
				children[1].weights[i] = p2.weights[i];
			}
			else {
				children[0].weights[i] = p2.weights[i];
				children[1].weights[i] = this.weights[i];
			}
		}
		return children;
	}

	// mutate an individual by altering one of its weights randomly
	public void mutate() {
		int MUTATED_WEIGHT = (int)Math.random() * NUM_WEIGHTS; // specific weight to mutate
		if (Math.random() < MUTATION_RATE) {
			if (MUTATED_WEIGHT == 1) {
				this.weights[MUTATED_WEIGHT] = (10)*Math.random(); // picked weight is linesCleared
			}
			else {
				this.weights[MUTATED_WEIGHT] = (-10)*Math.random(); // other weights picked
			}
		}
	}

	// allow sorting by gameScore
	@Override
	public int compareTo(Individual i) {
		return this.gameScore-i.gameScore;
	}
}

public class PlayerSkeleton{
	// create a population of size popSize
	public static Individual[] initializeRandomPopulation(int popSize) {
		Individual[] population = new Individual[popSize];
		for (int i = 0; i < popSize; i++) {
			population[i] = new Individual();
		}
		return population;
	}

	// returns sum of scores over NUM_GAMES games played
	// TODO: instead of for-loop, use mapreduce
	public static int getGameResult() {
		int NUM_GAMES=0; // number of games to play to determine an individual's gameScore
		int result = 0;
		for (int i = 0; i < NUM_GAMES; i++) {
			result += playGame();
		}
		return result;
	}

	// uses the genetic algorithm and returns the best weights
	public static double[] evolveWeights() {
		int POP_SIZE=0; // the size of the population
		int NUM_GENS=0; // the number of generations to evolve
		int REPLACEMENT_RATE=0; // proportion of population that will be replaced in the next generation
		int TOURNAMENT_RATE=0; // proportion of population that will be considered in each tournament
		int REPLACEMENT_SIZE = POP_SIZE * REPLACEMENT_RATE;
		int TOURNAMENT_SIZE = POP_SIZE * TOURNAMENT_RATE;

		Individual[] population = initializeRandomPopulation(POP_SIZE);
		// for every generation
		for (int i = 0; i < NUM_GENS; i++) {
			// play the game with current weights to obtain current fitness
			// iterating through population[] array
			// TODO: instead of iterating through population, use mapreduce
			for (int j = 0; j < POP_SIZE; j++) {
				population[j].gameScore = getGameResult();
			}

			// generate all the children for this generation
			Individual[] allChildren = new Individual[REPLACEMENT_SIZE];
			for (int j = 0; j < REPLACEMENT_SIZE/2; j++) {
				Individual[] tournamentPlayers = new Individual[TOURNAMENT_SIZE];
				Arrays.sort(tournamentPlayers);
				Individual p1 = tournamentPlayers[0];
				Individual p2 = tournamentPlayers[1];
				Individual[] children = p1.cross(p2);
				children[0].mutate();
				children[1].mutate();
				allChildren[i] = children[0];
				allChildren[i+1] = children[1];
			}

			// replace the weakest REPLACEMENT_SIZE individuals in the population with the children
			Arrays.sort(population); // strongest at front, weakest at back
			for (int j = POP_SIZE-(REPLACEMENT_SIZE+1); j < POP_SIZE-1; j++) {
				population[i] = allChildren[i];
			}
		}
		// return the weights of the strongest individual after evolution process is complete
		Arrays.sort(population);
		return population[0].weights;
	}

	// TODO
	public static double findFitness(int[][] nextState, int[] nextTop, double[] weights) {
		return 0;
	}

	// TODO
	public static int playGame() {
		State s = new State();
		new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while(!s.hasLost()) {
			s.makeMove(p.pickMove(s,s.legalMoves()));
			s.draw();
			s.drawNext(0,0);
			try {
				Thread.sleep(300);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed "+s.getRowsCleared()+" rows.");
		return s.getRowsCleared();
	}

	// TODO
	public static int pickMove(State s, int[][] legalMoves){
		return 0;
	}

	public static void main(String[] args) {
		if (args[0].equals("--evolve")) {
			double[] weights = evolveWeights();
			System.out.println("Evolved weights are" + Arrays.toString(weights));
		}
		// directly play game with found weights
		// TODO
		else {
			playGame();
		}
	}

}

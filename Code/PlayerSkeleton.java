import java.util.Arrays;

class Individual implements Comparable<Individual>{
	 double[] weights;
	 int NUM_WEIGHTS = 6;
	 int gameScore;

	public Individual(){
		weights = new double[NUM_WEIGHTS];
		for (int i = 0; i<NUM_WEIGHTS; i++){
			weights[i] = Math.random()*(-10);
		}
		weights[1] = -weights[1]; //weights[1] represents linesCleared

	}

	// cross two individuals to produce 2 children
	// to call: p1.cross(p2);
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

	// mutate an individual
	// to call: p1.mutate();
	public void mutate() {
		int MUTATION_RATE=0; // decide later
		int MUTATED_WEIGHT = (int)Math.random() * NUM_WEIGHTS; // specific weight to mutate
		if (Math.random() < MUTATION_RATE) {
			if (MUTATED_WEIGHT == 1) {
				this.weights[MUTATED_WEIGHT] = (10)*Math.random(); // picked weight is number of lines cleared
			}
			else {
				this.weights[MUTATED_WEIGHT] = (-10)*Math.random(); // other weights picked
			}
		}
	}

	// allow sorting by game score
	@Override
	public int compareTo(Individual i) {
		return this.gameScore-i.gameScore;
	}
}

public class PlayerSkeleton{
	public static Individual[] initializeRandomPopulation(int popSize) {
		Individual[] population = new Individual[popSize];
		for (int i = 0; i < popSize; i++) {
			population[i] = new Individual();
		}
		return population;
	}

	public static int getGameResult() {
		int NUM_GAMES=0;
		int result = 0;
		for (int i = 0; i < NUM_GAMES; i++) {
			result += playGame();
		}
		return result;
	}

	public static double[] evolveWeights() {
		int POP_SIZE=0; // the size of the population
		int NUM_GENS=0; // the number of generations to evolve
		int REPLACEMENT_RATE=0; // ex. 30%
		int TOURNAMENT_RATE=0; // ex. 10%
		int REPLACEMENT_SIZE = POP_SIZE * REPLACEMENT_RATE;
		int TOURNAMENT_SIZE = POP_SIZE * TOURNAMENT_RATE;

		Individual[] population = initializeRandomPopulation(POP_SIZE);
		// for every generation
		for (int i = 0; i < NUM_GENS; i++) {
			// play the game with current weights to obtain current fitness
			// iterating through population[] array
			for (int j = 0; j < POP_SIZE; j++) {
				population[j].gameScore = getGameResult();
			}

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

			Arrays.sort(population); // strongest at front, weakest at back
			for (int j = POP_SIZE-(REPLACEMENT_SIZE+1); j < POP_SIZE-1; j++) {
				population[i] = allChildren[i];
			}
		}
		Arrays.sort(population);
		return population[0].weights;
	}


	public static double findFitness(int[][] nextState, int[] nextTop, double[] weights) {
		// FIXME
		return 0;
	}

	// may need modifications
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


	public static int pickMove(State s, int[][] legalMoves){
		// FIXME
		return 0;
	}

	public static void main(String[] args) {
		// https://github.com/ngthnhan/Tetris/blob/final/src/PlayerSkeleton.java
		// perform evolution, or
		// perform game-playing
		// FIXME
	}

}
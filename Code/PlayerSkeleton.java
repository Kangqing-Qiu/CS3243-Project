import java.util.Arrays;

// class Individual contains methods related to individuals within a population
class Individual implements Comparable<Individual>{
	// number of heuristics used
	public static final int NUM_WEIGHTS = 6;
	// weight for each heuristic
	public double[] weights = new double[NUM_WEIGHTS];
	// result after playing games; equivalent to lines cleared after game
	public int gameScore = 0;

	// the constructor initializes the weights randomly. All weights are within
	// range (-10, 0) except for linesCleared, which is in the range (0, 10)
	// TESTED
	public Individual(){
		for (int i = 0; i<NUM_WEIGHTS; i++){
			weights[i] = Math.random()*(-10);
		}
		weights[1] = -weights[1]; //weights[1] represents linesCleared

	}

	// cross two individuals and returns the 2 children
	// TESTED
	public Individual[] cross(Individual p2, double crossRate) {
		Individual[] children = {this, p2};
		if (Math.random() > crossRate) {
			return children;
		}
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
	// TESTED
	// check if there's a better function to get a random int
	public void mutate(double mutationRate) {
		// chose a weight to mutate
		int MUTATED_WEIGHT = (int)(Math.random() * NUM_WEIGHTS);
		if (Math.random() < mutationRate) {
			if (MUTATED_WEIGHT == 1) { // picked weight is linesCleared
				this.weights[MUTATED_WEIGHT] = (10)*Math.random(); 
			}
			else {
				this.weights[MUTATED_WEIGHT] = (-10)*Math.random();
			}
		}
	}

	public static int getParentChildrenScoreDiff(
						Individual c1, Individual c2, Individual p1, 
						Individual p2) {
		return c1.gameScore + c2.gameScore - p1.gameScore - p2.gameScore;
	}

	// allow sorting by gameScore
	@Override
	public int compareTo(Individual i) {
		return this.gameScore-i.gameScore;
	}

	// for debugging purposes
	public void printInd() {
		System.out.println("Weights are: " + weights[0] + ", " + weights[1] + ", " + weights[2] + ", " + weights[3] + ", " + weights[4] + ", " + weights[5]);
		System.out.println("gameScore is: " + gameScore);
	}
}

// class Population contains methods related to operations on populations of individuals
class Population{
	// create a population of size popSize
	// TESTED
	public static Individual[] initializeRandomPopulation(int popSize) {
		Individual[] population = new Individual[popSize];
		for (int i = 0; i < popSize; i++) {
			population[i] = new Individual();
		}
		return population;
	}

	public static double getAvGameScore(Individual[] population, int size) {
		int gameScoreSum = 0;
		for (int i = 0; i < size; i++) {
			gameScoreSum += population[i].gameScore;
		}
		return 1.0 * gameScoreSum / size;
	}

	// for debugging purposes
	public static void printPop(Individual[] pop) {
		int len = pop.length;
		for (int i = 0; i < len; i++) {
			pop[i].printInd();
			System.out.println();
		}
	}
}

public class PlayerSkeleton{
	static int NUM_GENS = 2;
	// data for training/debugging/tuning purposes
	// 40 = number of generations
	// entry i is the data at the end of generation i
	static double[] scores = new double[NUM_GENS]; // average game scores
	static double[] crossRates = new double[NUM_GENS];
	static double[] mutationRates = new double[NUM_GENS];

	// returns sum of scores over NUM_GAMES games played
	// TODO: instead of for-loop, use mapreduce
	public static int getGameResult(double[] weights) {
		// number of games to play to determine an individual's gameScore
		int NUM_GAMES=0;
		int result = 0;
		for (int i = 0; i < NUM_GAMES; i++) {
			result += playGame(weights);
		}
		return result;
	}

	// returns an array of two elements that are the updated [cRate, mRate]
	public static double[] updateRates(double cRate, double mRate, 
		double cProgress, double mProgress, double cCount, 
		double mCount, int maxGS, int minGS, double avGS) {
		double delta;
		// if max and min are essentially the same
		if (maxGS - avGS < 0.00001) {delta = 0.01;} 
		else {delta = 0.01*(maxGS - avGS)/(maxGS - minGS);}

		double avCrossProgress = 1.0 * cProgress / cCount;
		double avMutationProgress = 1.0 * mProgress / mCount;
		// increase crossover rate, decrease mutation rate
		if (avCrossProgress > avMutationProgress) {
			cRate += delta;
			mRate -= delta;
		}
		// decrease crossover rate, increase mutation rate
		else {
			cRate -= delta;
			mRate += delta;
		}
		double[] rates = new double[2];
		rates[0] = cRate;
		rates[1] = mRate;
		return rates;
	}
	  
	// reduce the poplation size after half of the total generations
 	// half constant, half decreases exponentially
  	// NOT YET TESTED
  	public static void evolvePopSize(){
    		int t = 1;
    		if (GENS_count < NUM_GENS / 2){
      			POP_SIZE = POP_SIZE;
    		}
    		else{
      			POP_SIZE = (int)POP_SIZE * Math.exp(-t/5);
    		}
    		t++;
  	}
	
	// uses the genetic algorithm and returns the best weights
	public static double[] evolveWeights() {
		int POP_SIZE=10; // the size of the population
		// proportion of population to be replaced in next generation
		double REPLACEMENT_RATE=0.25;
		// proportion of population to be considered in each tournament
		double TOURNAMENT_RATE=0.5;
		// REPLACEMENT_SIZE must be even number
		int REPLACEMENT_SIZE = (int) Math.ceil(POP_SIZE * REPLACEMENT_RATE);
		if (REPLACEMENT_SIZE % 2 == 1) {REPLACEMENT_SIZE -= 1;}
		System.out.println("REPLACEMENT_SIZE: " + REPLACEMENT_SIZE);
		int TOURNAMENT_SIZE = (int) Math.ceil(POP_SIZE * TOURNAMENT_RATE);
		// (0,1) value that describes the chance with which a mutation occurs
		double mutationRate=(1.0/6.0);
		// (0,1) value that describes the chance with which crossover occurs
		double crossRate=0.5;

		// for debugging purposes
		int acceptCount = 0; 
		int rejectCount = 0;

		Individual[] population = Population.initializeRandomPopulation(POP_SIZE);
		// for every generation
		for (int i = 0; i < NUM_GENS; i++) {
			System.out.println("generation " + i);
			// how many children have been generated so far
			GENS_count = i;
      			// record the current generation number
			int childrenCount = 0;
			// number of crossovers performed in this generation
			int crossCount = 0;
			// sum of gameScore gain attributed to crossover
			int crossProgress = 0;
			// number of mutations performed in this generation
			int mutationCount = 0;
			// sum of gameScore gain attributed to mutation
			int mutationProgress = 0;
			// max, min, average gameScore in population
			int maxGameScore = 0;
			int minGameScore = 0;
			double avGameScore = 0.0;

			// play the game with current weights to obtain current fitness
			// iterating through population[] array
			// TODO: instead of iterating through population, use mapreduce
			for (int j = 0; j < POP_SIZE; j++) {
				population[j].gameScore = getGameResult(population[j].weights);
				evolvePopSize();
			}

			// generate all the children for this generation
			Individual[] allChildren = new Individual[REPLACEMENT_SIZE];
			int childIndex = 0;
			while (childrenCount < REPLACEMENT_SIZE) {
				boolean crossed = false;
				boolean mutatedOne = false;
				boolean mutatedTwo = false;

				Individual[] tournamentPlayers = new Individual[TOURNAMENT_SIZE];
				// randomly choose TOURNAMENT_SIZE number of individuals
				for (int k = 0; k < TOURNAMENT_SIZE; k++) {
					int chosen = (int) (Math.random() * POP_SIZE); // check off by one?
					tournamentPlayers[k] = population[chosen];
				}
				Arrays.sort(tournamentPlayers);
				Individual p1 = tournamentPlayers[0];
				Individual p2 = tournamentPlayers[1];
				Individual[] children = {p1, p2};
				if (Math.random() < crossRate) {
					children = p1.cross(p2, crossRate);
					crossCount++;
					crossed = true;

					children[0].gameScore = getGameResult(children[0].weights);
					children[1].gameScore = getGameResult(children[1].weights);

					// discard children if they are both worse than the worst parent
					// otherwise, retain both children
					if (children[0].gameScore < p2.gameScore 
						&& children[1].gameScore < p2.gameScore) {
						System.out.println("rejected crossed children");
						rejectCount++;
						if (crossed) {crossCount--;}
						continue; // go back to while loop
					}

					crossProgress += Individual.getParentChildrenScoreDiff(
						children[0], children[1], p1, p2);
				}

				if (Math.random() < mutationRate) {
					children[0].mutate(mutationRate);
					mutationCount++;
					mutatedOne = true;
				}
				if (Math.random() < mutationRate) {
					children[1].mutate(mutationRate);
					mutationCount++;
					mutatedTwo = true;
				}

				if (mutatedOne || mutatedTwo) {
					if (mutatedOne) {children[0].gameScore = getGameResult(children[0].weights);}
					if (mutatedTwo) {children[1].gameScore = getGameResult(children[1].weights);}
					
					// discard children if they are both worse than the worst parent
					// otherwise, retain both children
					if (children[0].gameScore < p2.gameScore && children[1].gameScore < p2.gameScore) {
						System.out.println("rejected mutated children");
						rejectCount++;
						if (mutatedOne) {mutationCount--;}
						if (mutatedTwo) {mutationCount--;}
						continue; // go back to while loop
					}

					if (mutatedOne && mutatedTwo) {
						mutationProgress += (2 * Individual.getParentChildrenScoreDiff(children[0], children[1], p1, p2));
					}
					else {
						mutationProgress += Individual.getParentChildrenScoreDiff(children[0], children[1], p1, p2);
					}
				}
				System.out.println("accepting children");
				acceptCount++;
				childrenCount += 2;

				allChildren[childIndex] = children[0];
				allChildren[childIndex+1] = children[1];
				childIndex += 2;
			}
			System.out.println("printing allChildren");
			System.out.println("length is " + REPLACEMENT_SIZE);
			Population.printPop(allChildren);
			System.out.println("finished creating all children");

			// can we avoid doing this here? (need allChildren[j] to have gameScore)
			for (int j = 0; j < REPLACEMENT_SIZE; j++) {
				allChildren[j].gameScore = getGameResult(allChildren[j].weights);
			}
			Arrays.sort(allChildren); 
			// replace the weakest REPLACEMENT_SIZE individuals in the population with the children
			Arrays.sort(population); // strongest at front, weakest at back
			for (int j = POP_SIZE-REPLACEMENT_SIZE; j < POP_SIZE; j++) {
				population[j] = allChildren[j-(POP_SIZE-REPLACEMENT_SIZE)];
			}

			// adjust crossover and mutation rates for next generation
			maxGameScore = Math.max(
							population[0].gameScore, 
							allChildren[0].gameScore);
			minGameScore = Math.min(
							population[POP_SIZE-REPLACEMENT_SIZE].gameScore, 
							allChildren[REPLACEMENT_SIZE-1].gameScore);
			avGameScore = Population.getAvGameScore(population, POP_SIZE);
			double[] newRates = updateRates(crossRate, mutationRate, 
											crossProgress, mutationProgress,
											crossCount, mutationCount, 
											maxGameScore, minGameScore, 
											avGameScore);
			crossRate = newRates[0];
			mutationRate = newRates[1];

			// keep track of debugging data
			scores[i] = avGameScore;
			crossRates[i] = crossRate;
			mutationRates[i] = mutationRate;
		}
		System.out.println("rejected " + rejectCount + ", accepted " + acceptCount);
		// return the weights of the strongest individual after evolution process is complete
		Arrays.sort(population);
		return population[0].weights;
	}

	// TODO
	public static double findFitness(int[][] nextState, int[] nextTop, double[] weights) {
		return 0;
	}

	// TODO
	// note: pickMove should take weights into consideration
	public static int playGame(double[] weights) {
		return (int) (Math.random() * 50);
	}

	// TODO
	public static int pickMove(State s, int[][] legalMoves){
		return 0;
	}

	public static void plotData(String name, double[] data) {
		System.out.println("Graph for " + name + ":");
		for (int i = 0; i < data.length; i++) {
			System.out.print(data[i] + " ");
		}
		double[] x = new double[data.length]; x[0] = 1;
		for(int i = 1; i < x.length; i++){
			x[i] = i+1;
		}
		MatlabChart fig = new MatlabChart();
		fig.plot(x, data, "-r", 2.0f, "AAPL");
		fig.RenderPlot();
		fig.title(name);
		fig.xlim(10, 100);
		fig.ylim(200, 300);
		fig.xlabel("Days");
		fig.ylabel("Price");
		fig.grid("on","on");
		fig.legend("northeast");
		fig.font("Helvetica",15);
		fig.saveas(name+".jpeg",640,480);
	}

	public static void runTests() {
		Individual[] pop = Population.initializeRandomPopulation(5);
		// Population.printPop(pop);

		Individual[] children = pop[0].cross(pop[1], 1.0);
		Population.printPop(children); 

		children[0].mutate(1.0);
		children[0].printInd();

		double[] data = {9.0, 4.0, 3.0};
		plotData("test", data);

	}

	public static void main(String[] args) {
		double[] foundWeights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
		switch (args[0]) {
			case "--evolve":
				double[] weights = evolveWeights();
				System.out.println("Evolved weights are" + Arrays.toString(weights));
				//plotData for scores, crossRates and mutationRates vs gen
				plotData("scores vs gen", scores);
				plotData("crossRates vs gen", crossRates);
				plotData("mutationRates vs gen", mutationRates);
				break;
			// directly play game with found weights
			// TODO
			case "--play":
				playGame(foundWeights);
				break;
			case "--test":
				runTests();
				break;
			default:
				System.out.println("To evolve the weights, use '--evolve'.");
				System.out.println("To play the game using our weights, use '--play'.");
				System.out.println("To run tests, use '--test'.");
				break;
		}
	}

}

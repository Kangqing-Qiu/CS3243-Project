import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.*;

// class Individual contains methods related to individuals within a population
// TESTED all methods in this class
class Individual implements Comparable<Individual>{
	// number of heuristics used
	public static final int NUM_WEIGHTS = 6;
	// weight for each heuristic
	public double[] weights = new double[NUM_WEIGHTS];
	// result after playing games; equivalent to lines cleared after game
	// TODO: double check whether it is consistent that this describes the score of one game or sum over NUM_GAMES
	public int gameScore = 0;

	// the constructor initializes the weights randomly. All weights are within
	// range (-10, 0) except for linesCleared, which is in the range (0, 10)
	public Individual(){
		for (int i = 0; i<NUM_WEIGHTS; i++){
			weights[i] = Math.random()*(-10);
		}
		weights[1] = -weights[1]; //weights[1] represents linesCleared
	}

	public Individual(double[] weights, int score) {
		for (int i = 0; i<NUM_WEIGHTS; i++){
			this.weights[i] = weights[i];
		}
		this.gameScore = score;
	}

	// return a copy of the given individual
	public static Individual returnCopy(Individual ind) {
		double[] weights = new double[NUM_WEIGHTS];
		for (int i = 0; i < NUM_WEIGHTS; i++) {
			weights[i] = ind.weights[i];
		}
		int score = ind.gameScore;
		Individual copy = new Individual(weights, score);
		return copy;
	}

	// cross two individuals and return the 2 children
	public Individual[] cross(Individual p2) {
		Individual[] children = new Individual[2];
		children[0] = returnCopy(this);
		children[1] = returnCopy(p2);

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
		// chose a weight to mutate
		int MUTATED_WEIGHT = (int)(Math.random() * NUM_WEIGHTS);
		if (MUTATED_WEIGHT == 1) { // picked weight is linesCleared
			this.weights[MUTATED_WEIGHT] = (10) * Math.random();
		}
		else {
			this.weights[MUTATED_WEIGHT] = (-10)*Math.random();
		}
	}

	/*
	// this method does not seem to improve performance
	public void mutateDynamically(int currGen, int numGens) {
		// first half of generations mutate completely randomly
		if (currGen < numGens / 2) this.mutate();
		else {
			// chose a weight to mutate and choose a percentage change
			int MUTATED_WEIGHT = (int)(Math.random() * NUM_WEIGHTS);
			double[] percentages = {0.1, 0.15};
			int index = (int) (Math.random() * 2);
			double percent = percentages[index];

			boolean negative = (this.weights[MUTATED_WEIGHT] < 0);
			double delta = Math.abs(this.weights[MUTATED_WEIGHT] * percent);
			// increase by percent
			if (Math.random() > 0.5) {
				double newVal = this.weights[MUTATED_WEIGHT] + delta;
				if (negative && (newVal > -0.0001))
					this.weights[MUTATED_WEIGHT] = -0.0001;
				else if (!negative && (newVal > 9.9999))
					this.weights[MUTATED_WEIGHT] = 9.9999;
				else this.weights[MUTATED_WEIGHT] = newVal;
			}
			// decrease by percent
			else {
				double newVal = this.weights[MUTATED_WEIGHT] - delta;
				if (negative && (newVal < -9.9999))
					this.weights[MUTATED_WEIGHT] = -9.9999;
				else if (!negative && (newVal < 0.0001))
					this.weights[MUTATED_WEIGHT] = 0.0001;
				else this.weights[MUTATED_WEIGHT] = newVal;
			}
		}
	}
	*/

	public static int getParentChildrenScoreDiff(
						Individual c1, Individual c2, Individual p1,
						Individual p2) {
		return c1.gameScore + c2.gameScore - p1.gameScore - p2.gameScore;
	}

	// allow sorting by gameScore: strongest at front, weakest at back
	@Override
	public int compareTo(Individual i) {
		return i.gameScore-this.gameScore;
	}

	// for debugging purposes; print an individual
	public void printInd() {
		System.out.println("Weights are: " + weights[0] + ", " + weights[1] + ", " + weights[2] + ", " + weights[3] + ", " + weights[4] + ", " + weights[5]);
		System.out.println("gameScore is: " + gameScore);
	}
}

// class Population contains methods related to operations on populations of individuals
// TESTED all methods in this class
class Population{
	// create a population of size popSize
	public static Individual[] initializeRandomPopulation(int popSize) {
		Individual[] population = new Individual[popSize];
		for (int i = 0; i < popSize; i++) {
			population[i] = new Individual();
		}
		return population;
	}

	// calculate the average game score of individuals in a population.
	public static double getAvGameScore(Individual[] population) {
		int size = population.length;
		int gameScoreSum = 0;
		for (int i = 0; i < size; i++) {
			gameScoreSum += population[i].gameScore;
		}
		return 1.0 * gameScoreSum / size;
	}

	// calculate the variance of game scores in a population.
	public static double getVariance(double ave, Individual[] population) {
		int size = population.length;
		double sum = 0;
		for (int i = 0; i < size; i++) {
			sum += Math.pow((population[i].gameScore - ave), 2);
		}
		return sum / size;

	}

	// for debugging purposes; print all individuals in a population
	public static void printPop(Individual[] pop) {
		for (Individual aPop : pop) {
			aPop.printInd();
			System.out.println();
		}
	}
}

// class PlayOnThisBoard contains methods to duplicate a given board and make
// moves on the new copy of the board.
class PlayOnThisBoard{
	public int[][] playField;
	public int[] playTop;
	public PlayOnThisBoard(int[][] oldField,int[] oldTop){
		playTop = Arrays.copyOf(oldTop, oldTop.length);
		playField = new int[oldField.length][];
		for(int i = 0; i < oldField.length; i++){
			playField[i] = Arrays.copyOf(oldField[i], oldField[i].length);
		}
	}

	public int[][] getPlayField(){
		return playField;
	}

	public int[] getPlayTop(){
		return playTop;
	}

	// similar to makeMove in State.java, since we cannot modify the original
	// board each time we evaluate a move
	public Boolean playMove(State s, int orient, int slot) {
		int pWidth[][] = s.getpWidth();
		int pHeight[][] = s.getpHeight();
		int pTop[][][] = s.getpTop();
		int pBottom[][][] = s.getpBottom();
		int nextPiece = s.getNextPiece();
		int turnNumber = s.getTurnNumber();
		turnNumber++;
		//height if the first column makes contact
		int height = playTop[slot]-pBottom[nextPiece][orient][0];
		//for each column beyond the first in the piece
		for (int c = 1; c < pWidth[nextPiece][orient]; c++) {
			height = Math.max(height,playTop[slot+c] - pBottom[nextPiece][orient][c]);
		}
		//check if game ended
		if (height+pHeight[nextPiece][orient] >= State.ROWS) {
			return false;
		}
		//for each column in the piece - fill in the appropriate blocks
		for (int i = 0; i < pWidth[nextPiece][orient]; i++) {
			//from bottom to top of brick
			for (int h = height+pBottom[nextPiece][orient][i]; h < height+pTop[nextPiece][orient][i]; h++) {
				playField[h][i+slot] = turnNumber;
			}
		}
		//adjust top
		for (int c = 0; c < pWidth[nextPiece][orient]; c++) {
			playTop[slot+c] = height+pTop[nextPiece][orient][c];
		}
		return true;
	}
}

public class PlayerSkeleton{
	public static class evolveParallelism implements Runnable{
		int single;
		int POP_SIZE;
		Individual[] population;
		CountDownLatch cdl;
		evolveParallelism(Individual[] population, int single, int POP_SIZE, CountDownLatch cdl){
			this.single = single;
			this.population = population;
			this.POP_SIZE = POP_SIZE;
			this.cdl = cdl;
		}
		@Override
		public void run() {
			population[single].gameScore = getGameResult(population[single].weights);
			cdl.countDown();
			System.out.println("Waiting for: "+cdl.getCount()+" threads to finish");
		}
	}
	public static class gameParallelism implements Callable<Integer> {
		double[] weights;
		int result;
		gameParallelism(double[] weights, int result){
			this.weights = weights;
			this.result = result;
		}
		@Override
		public Integer call() throws Exception {
			try{
				result = playGame(weights);
			}catch (Exception e){
				e.printStackTrace();
			}
			return result;
		}
	}

	static int NUM_GENS = 30;
	// data for training/debugging/tuning purposes
	// entry i is the data at the end of generation i
	static double[] scores = new double[NUM_GENS]; // average game scores
	static double[] crossRates = new double[NUM_GENS];
	static double[] mutationRates = new double[NUM_GENS];
	static double[] deltas = new double[NUM_GENS];

	// returns average of scores over NUM_GAMES games played
	public static int getGameResult(double[] weights) {
		// number of games to play to determine an individual's gameScore
		int NUM_GAMES=3;
		int result = 0;
		int threads = Runtime.getRuntime().availableProcessors();
		ExecutorService executor = Executors.newFixedThreadPool(threads);
		ArrayList<Future<Integer>> futures = new ArrayList<>();
		for(int i = 0; i < NUM_GAMES; i++) {
			Callable<Integer> gameParallelism = new gameParallelism(weights, result);
			//System.out.println("Individual "+ Arrays.toString(weights) +" task "+i+" is executing");
			Future<Integer> future = executor.submit(gameParallelism);
			futures.add(future);
		}
		executor.shutdown();
		for(Future<Integer> future : futures){
			try {
				result+=future.get();
			} catch (InterruptedException | ExecutionException e) {
				e.printStackTrace();
			}
		}
		//System.out.println("All threads are done for weights "+ Arrays.toString(weights));
		//System.out.println("Individual with weights "+ Arrays.toString(weights) +" gets "+result);
		return (int) (1.0 * result / NUM_GAMES);
	}

	// returns an array of two elements that are the updated [cRate, mRate]
	// TESTED
	public static double[] updateRates(int i, double cRate, double mRate,
		int cProgress, int mProgress, int cCount,
		int mCount, double avGS, double variance) {
		double delta = 100 / getDispersion(avGS, variance);
		deltas[i] = delta;
		System.out.println("delta is " + delta);
		double avCrossProgress = 0.0;
		if (cCount != 0) avCrossProgress = 1.0 * cProgress / cCount;
		double avMutationProgress = 0.0;
		if (mCount != 0) avMutationProgress = 1.0 * mProgress / mCount;
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

	// reduce the population size after half of the total generations
	// population size decreases exponentially but maintains threshold of 10
	// note: 20 because tournament size = 20 * 0.1 = 2, minimum required.
	// TESTED
  	public static int evolvePopSize(int genCount, int size){
		System.out.println("evolvePopSize(" + genCount + ", " + size + ")");
		boolean decrease = true;
		int newSize = size;
		if (genCount >= NUM_GENS / 2 && decrease) {
			newSize = (int) (size * Math.exp(-1.0 / 20));
			if (newSize < 41) {
				newSize = size;
				decrease = false;
			}
        }
		return newSize;
  	}

  	// adjust replacement size to a new popSize
  	public static int scaleReplacementSize(int popSize, double rate) {
		System.out.println("scaleReplacementSize(" + popSize + ", " + rate + ")");
		int newSize = (int) Math.ceil(popSize * rate);
		if (newSize % 2 == 1) {newSize -= 1;}
		return newSize;
	}

	// adjust tournament size to a new popSize
	public static int scaleTournamentSize(int popSize, double rate) {
		System.out.println("scaleTournamentSize(" + popSize + ", " + rate + ")");
		return (int) Math.ceil(popSize * rate);
	}

	// calculate the index of dispersion with the given average and variance
	public static double getDispersion(double ave, double variance) {
		System.out.println("getDispersion(" + ave + ", " + variance + ")");
		return variance / ave;
	}

	// uses the genetic algorithm and returns the best weights
	public static double[] evolveWeights() {
		int POP_SIZE=100; // the size of the population
		// proportion of population to be replaced in next generation
		double REPLACEMENT_RATE=0.25;
		// proportion of population to be considered in each tournament
		double TOURNAMENT_RATE=0.1;
		// for convenience, REPLACEMENT_SIZE must be even number since we
		// produce either 0 or 2 children each time
		int REPLACEMENT_SIZE = 0;
		int TOURNAMENT_SIZE = 0;
		// (0,1) value that describes the chance with which a mutation occurs
		double mutationRate=(1.0/6.0);
		// (0,1) value that describes the chance with which crossover occurs
		double crossRate=0.9;

		Individual[] population = Population.initializeRandomPopulation(POP_SIZE);
		Individual[] newPopulation; // auxiliary array used to adjust population size
		// for every generation
		for (int i = 0; i < NUM_GENS; i++) {
			System.out.println("generation " + i);
			// how many children have been generated so far
			int childrenCount = 0;
			// number of crossovers performed in this generation
			int crossCount = 0;
			// sum of gameScore gain attributed to crossover
			int crossProgress = 0;
			// number of mutations performed in this generation
			int mutationCount = 0;
			// sum of gameScore gain attributed to mutation
			int mutationProgress = 0;
			// max, average, variance of gameScore in population
			int maxGameScore = 0;
			double avGameScore = 0.0;
			double variance = 0.0;

			// play the game with current weights to obtain current fitness
			int threads = POP_SIZE;
			ExecutorService executor = Executors.newFixedThreadPool(threads);
			CountDownLatch cdl = new CountDownLatch(threads);
			for (int j = 0; j < POP_SIZE; j++) {
				System.out.println("Individual "+ Arrays.toString(population[j].weights) +" is executing");
				Runnable individual = new evolveParallelism(population,j,POP_SIZE,cdl);
				executor.execute(individual);
			}
			System.out.println("now all threads are executed, waiting...");
			try {
				cdl.await();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.out.println("now all done.");
			executor.shutdown();

			// adjust population and batch sizes
			int oldPopSize = POP_SIZE;
			POP_SIZE = evolvePopSize(i,POP_SIZE);
			REPLACEMENT_SIZE = scaleReplacementSize(POP_SIZE, REPLACEMENT_RATE);
			TOURNAMENT_SIZE = scaleTournamentSize(POP_SIZE, TOURNAMENT_RATE);

			// generate all the children for this generation
			Individual[] allChildren = new Individual[REPLACEMENT_SIZE];
			int childIndex = 0;
			// TODO: since we now call this a fixed number of times, can parallelize
			// ie we call it (REPLACEMENT_SIZE / 2) times
			while (childrenCount < REPLACEMENT_SIZE) {
				boolean crossed = false;
				boolean mutatedOne = false;
				boolean mutatedTwo = false;

				// randomly choose TOURNAMENT_SIZE number of individuals
				Individual[] tournamentPlayers = new Individual[TOURNAMENT_SIZE];
				for (int k = 0; k < TOURNAMENT_SIZE; k++) {
					int chosen = (int) (Math.random() * POP_SIZE);
					tournamentPlayers[k] = population[chosen];
				}
				Arrays.sort(tournamentPlayers);
				System.out.println("Tournament size = " + POP_SIZE  + " * " + TOURNAMENT_RATE + " ~approx " + TOURNAMENT_SIZE);
				System.out.println("Tournament players are: ");
				Population.printPop(tournamentPlayers);
				Individual p1 = tournamentPlayers[0];
				Individual p2 = tournamentPlayers[1];
				Individual[] children = new Individual[2];
				children[0] = Individual.returnCopy(p1);
				children[1] = Individual.returnCopy(p2);
				if (Math.random() < crossRate) {
					children = p1.cross(p2);
					crossCount++;
					crossed = true;

					// calculated childrens' new gameScores
					children[0].gameScore = getGameResult(children[0].weights);
					children[1].gameScore = getGameResult(children[1].weights);

					crossProgress += (2 * Individual.getParentChildrenScoreDiff(children[0], children[1], p1, p2));
				}

				if (Math.random() < mutationRate) {
					children[0].mutate();
					mutationCount++;
					mutatedOne = true;
				}
				if (Math.random() < mutationRate) {
					children[1].mutate();
					mutationCount++;
					mutatedTwo = true;
				}

				if (mutatedOne || mutatedTwo) {
					if (mutatedOne) {children[0].gameScore = getGameResult(children[0].weights);}
					if (mutatedTwo) {children[1].gameScore = getGameResult(children[1].weights);}

					if (mutatedOne && mutatedTwo) {
						mutationProgress += (2 * Individual.getParentChildrenScoreDiff(children[0], children[1], p1, p2));
					}
					else {
						mutationProgress += Individual.getParentChildrenScoreDiff(children[0], children[1], p1, p2);
					}
				}

				childrenCount += 2;
				allChildren[childIndex] = children[0];
				allChildren[childIndex+1] = children[1];
				childIndex += 2;
				System.out.println("children from this tournament:");
				Population.printPop(children);
				System.out.println();
			}
			System.out.println("printing allChildren");
			System.out.println("length is " + REPLACEMENT_SIZE);
			Population.printPop(allChildren);
			System.out.println("finished creating all children");
			System.out.println("(mutated, crossed) = (" + mutationCount + ", " + crossCount + ")");

			Arrays.sort(allChildren);

			// discard weakest of population to scale down to new population size
			// then, replace the next weakest REPLACEMENT_SIZE individuals in the population with the children
			Arrays.sort(population); // strongest at front, weakest at back
			if (oldPopSize != POP_SIZE) {
				newPopulation = new Individual[POP_SIZE];
				for (int k = 0; k < POP_SIZE; k++) {
					newPopulation[k] = population[k];
				}
				population = newPopulation;
			}
			for (int j = POP_SIZE-REPLACEMENT_SIZE; j < POP_SIZE; j++) {
				population[j] = allChildren[j-(POP_SIZE-REPLACEMENT_SIZE)];
			}

			// adjust crossover and mutation rates for next generation
			maxGameScore = Math.max(
							population[0].gameScore, 
							allChildren[0].gameScore);
			System.out.println("end of gen " + i + ": best parent has score " + population[0].gameScore + " and weights = " + Arrays.toString(population[0].weights));
			System.out.println("end of gen " + i + ": best child weights = " + allChildren[0].gameScore + " and weights = " + Arrays.toString(allChildren[0].weights));
			avGameScore = Population.getAvGameScore(population);
			variance = Population.getVariance(avGameScore, population);
			double[] newRates = updateRates(i, crossRate, mutationRate,
											crossProgress, mutationProgress,
											crossCount, mutationCount,
											avGameScore, variance);

			// sanity check
			if (maxGameScore < avGameScore) {System.out.println("FAILURE: maxGS < avGS");}


			crossRate = newRates[0];
			mutationRate = newRates[1];


			// keep track of debugging data
			scores[i] = avGameScore;
			crossRates[i] = crossRate;
			mutationRates[i] = mutationRate;
			System.out.println("end of gen " + i + ": avGameScore = " + avGameScore);
			System.out.println("end of gen " + i + ": crossRate = " + crossRate + ", mutationRate = " + mutationRate);
		}
		// return the weights of the strongest individual after evolution process is complete
		Arrays.sort(population);
		return population[0].weights;
	}

	//the final keyword is used in several contexts to define an entity that can only be assigned once.
	public double findFitness(final int[][] playField, final int[] playTop,double[] tempWgts){
		int maxRow = playField.length;
		int maxCol = playField[0].length;
		// features
		double landingHeight = 0;
		double rowsCleared = 0;
		double rowTransitions = 0;
		double columnTransitions = 0;
		double numHoles = 0;
		double wellSums = 0;
		int moveNumber = -1;

		// calculate landingHeight
		for(int i = 0; i<maxCol; i++) {
			for (int j  = playTop[i]-1; j >=0; j--) {
				if(playField[j][i] == 0) numHoles++;
			}
			// System.out.println(Math.max(newTop[i]-1, 0));
			if(playField[Math.max(playTop[i]-1, 0)][i] > moveNumber) {
				moveNumber = playField[Math.max(playTop[i]-1, 0)][i];

				landingHeight = playTop[i];
			}
		}

		// calculate rowTransitions and rowsCleared
		for(int i = 0; i<maxRow; i++) {
			boolean lastCell = false;
			boolean currentCell = false;
			int rowIsClear = 1;
			for (int j = 0; j<maxCol; j++) {
				currentCell = false;
				if(playField[i][j] == 0) {
					rowIsClear = 0;
					currentCell = true;
				}

				if(lastCell != currentCell) {
					rowTransitions++;
				}
				lastCell = currentCell;
			}
			rowsCleared+=rowIsClear;
			if(currentCell) rowTransitions++;
		}

		// calculate columnTransitions
		for(int i = 0; i<maxCol; i++) {
			boolean lastCell = true;
			boolean currentCell = false;
			for (int j = 0; j<maxRow-1; j++) {
				currentCell = (playField[j][i] != 0);
				// if(!currentCell && newField[j+1][i] !=0) numHoles++;
				if(lastCell != currentCell) {
					columnTransitions++;
				}
				lastCell = currentCell;
			}
			// if(!currentCell) columnTransitions++;
		}

		// calculate wellSums
		for(int i = 1; i<maxCol-1; i++) {
			for(int j = 0; j < maxRow; j++) {
				if(playField[j][i] == 0 && playField[j][i-1] != 0 && playField[j][i+1] != 0) {
					wellSums++;
					for (int k = j -1; k >=0; k--) {
						if(playField[k][i] == 0) wellSums++;
						else break;
					}
				}
			}
		}
		for(int j = 0; j < maxRow; j++) {
			if(playField[j][0] == 0 && playField[j][1] != 0) {
				wellSums++;
				for (int k = j-1; k >= 0; k--) {
					if (playField[k][0] == 0) wellSums++;
					else break;
				}
			}
			if(playField[j][maxCol-1] == 0 && playField[j][maxCol-2] != 0) {
				wellSums++;
				for (int k = j-1; k >= 0; k--) {
					if (playField[k][maxCol-1] == 0) wellSums++;
					else break;
				}
			}
		}

		return landingHeight*tempWgts[0] + rowsCleared*tempWgts[1]+ rowTransitions*tempWgts[2] +
				columnTransitions*tempWgts[3] + numHoles*tempWgts[4] + wellSums*tempWgts[5];
	}

	public static int playGame(double[] weights) {
		State s = new State();
		//new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while(!s.hasLost()) {
			s.makeMove(p.pickMove(weights, s, s.legalMoves()));   //make this optimal move
			//s.draw();
			//s.drawNext(0,0);
			/*
			try {
				Thread.sleep(300);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			*/
		}
		//System.out.println(s.getRowsCleared());
		System.out.println("You have completed "+s.getRowsCleared()+" rows with weights"+ Arrays.toString(weights));
		return s.getRowsCleared();
	}

	public int pickMove(double[] weights, State s, int[][] legalMoves) {
		double maxScore = Integer.MIN_VALUE;
		int optimalMove = Integer.MIN_VALUE;
		int[] oldTop = s.getTop();
		int[][] oldField = s.getField();

		for(int moveCount = 0; moveCount < legalMoves.length; moveCount++) {
			int orient = legalMoves[moveCount][0];
			int slot = legalMoves[moveCount][1];

			// evaluate this move on a copy of the given board
			PlayOnThisBoard playBoard = new PlayOnThisBoard(oldField,oldTop);
			if(playBoard.playMove(s, orient, slot)){
				double tempScore = findFitness(playBoard.getPlayField(), playBoard.getPlayTop(), weights);
				//whenever the score is similar,random check update or not
				if(Math.abs(tempScore - maxScore) < 0.000000001){
					if(Math.random() > 0.5)
						optimalMove = moveCount;
				}
				//if significantly improved,update
				else if(tempScore > maxScore){
					optimalMove = moveCount;
					maxScore = tempScore;
				}
			}
		}
		if (optimalMove == Integer.MIN_VALUE) {return 0;}
		return optimalMove;
	}
/*
	// TESTED
	public static void plotData(String name, String x, String y, int ymax, double[] data) {
		System.out.println("Graph for " + name + ":");
		for (double aData : data) {
			System.out.print(aData + " ");
		}
		System.out.println();

		double[] xa = new double[data.length]; xa[0] = 1;
		for(int i = 1; i < xa.length; i++){
			xa[i] = i+1;
		}
		MatlabChart fig = new MatlabChart();

		fig.plot(xa, data, "-r", 2.0f, "data");
		fig.RenderPlot();
		fig.title(name);
		fig.xlim(0, 100);
		fig.ylim(0, ymax);
		fig.xlabel(x);
		fig.ylabel(y);
		fig.grid("on","on");
		fig.legend("northeast");
		fig.font("Helvetica",15);
		fig.saveas(name+".jpeg",640,480);
	}
*/
	public static void runTests() {
		Individual[] pop = Population.initializeRandomPopulation(5);
		pop[0].gameScore = 3;
		pop[1].gameScore = 6;
		pop[2].gameScore = 2;
		pop[3].gameScore = 4;
		Arrays.sort(pop);
		System.out.println("ave = " + Population.getAvGameScore(pop));
		//Population.printPop(pop);

		Individual[] children = pop[0].cross(pop[1]);
		// Population.printPop(children);

		children[0].mutate();
		//children[0].printInd();

		// double[] data = {100.0, 50.0, 20.0, 80.0};
		// plotData("test", "t", "val", data);

		int currSize = 100;
		for (int i = 0; i < NUM_GENS; i++) {
			int nextSize = evolvePopSize(i, currSize);
			currSize = nextSize;
			System.out.println("currSize is " + currSize);
		}

		/*
		for (int i = 0; i < 10; i++) {
			children[0].mutateDynamically(i, 10);
			children[0].printInd();
		}*/
	}

	public static void main(String[] args) {
		//double[] foundWeights = {-7.25,3.87,-7.25,-7.25,-7.25,-7.25};
		double[] foundWeights = {-2.5953031561074615, 6.135171396733583, -2.184882625105884, -5.9874618089311396, -7.098554416480493, -2.4152223172808496};
		switch (args[0]) {
			case "--evolve":
				double[] weights = evolveWeights();
				System.out.println("Evolved weights are" + Arrays.toString(weights));
				//plotData("scores vs gen", "gen", "score", 5000, scores);
				//plotData("crossRates vs gen", "gen", "crossRate", 1, crossRates);
				//plotData("mutationRates vs gen", "gen", "mutationRate", 1, mutationRates);
				//plotData("deltas vs gen", "gen", "delta", 1, deltas);
				break;
			case "--play":
				/*
				int NUM_GAMES = 100;
				int result = 0;
				int threads = Runtime.getRuntime().availableProcessors();
				ExecutorService executor = Executors.newFixedThreadPool(threads);
				for (int i = 0; i < NUM_GAMES; i++) {
					Callable<Integer> gameParallelism = new gameParallelism(foundWeights, result);
					executor.submit(gameParallelism);
				}
				executor.shutdown();
				*/
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

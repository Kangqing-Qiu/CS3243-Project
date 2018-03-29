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
	public int gameScore = 0;

	// the constructor initializes the weights randomly. All weights are within
	// range (-10, 0) except for linesCleared, which is in the range (0, 10)
	public Individual(){
		for (int i = 0; i<NUM_WEIGHTS; i++){
			weights[i] = Math.random()*(-10);
		}
		weights[1] = -weights[1]; //weights[1] represents linesCleared

	}

	// cross two individuals and return the 2 children
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

	// allow sorting by gameScore: smallest to largest
	@Override
	public int compareTo(Individual i) {
		return this.gameScore-i.gameScore;
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

	// for debugging purposes; print all individuals in a population
	public static void printPop(Individual[] pop) {
		for (Individual aPop : pop) {
			aPop.printInd();
			System.out.println();
		}
	}
}

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

	// similar to makemove in State.java, since we cannot do this on the original board
	public Boolean playMove(State s,int orient,int slot) {
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
		for(int c = 1; c < pWidth[nextPiece][orient]; c++) {
			height = Math.max(height,playTop[slot+c] - pBottom[nextPiece][orient][c]);
		}
		//check if game ended
		if(height+pHeight[nextPiece][orient] >= State.ROWS) {
			return false;
		}
		//for each column in the piece - fill in the appropriate blocks
		for(int i = 0; i < pWidth[nextPiece][orient]; i++) {
			//from bottom to top of brick
			for(int h = height+pBottom[nextPiece][orient][i]; h < height+pTop[nextPiece][orient][i]; h++) {
				playField[h][i+slot] = turnNumber;
			}
		}
		//adjust top
		for(int c = 0; c < pWidth[nextPiece][orient]; c++) {
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
		evolveParallelism(Individual[] population, int single, int POP_SIZE){
			this.single = single;
			this.population = population;
			this.POP_SIZE = POP_SIZE;
		}
		@Override
		public void run() {
			population[single].gameScore = getGameResult(population[single].weights);
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

	static int NUM_GENS = 50;
	// data for training/debugging/tuning purposes
	// 40 = number of generations
	// entry i is the data at the end of generation i
	static double[] scores = new double[NUM_GENS]; // average game scores
	static double[] crossRates = new double[NUM_GENS];
	static double[] mutationRates = new double[NUM_GENS];
 	static int t = 1; // time variable

	// returns sum of scores over NUM_GAMES games played
	public static int getGameResult(double[] weights) {
		// number of games to play to determine an individual's gameScore
		int NUM_GAMES=5;
		int result = 0;
		int threads = Runtime.getRuntime().availableProcessors();
		ExecutorService executor = Executors.newFixedThreadPool(threads);
		ArrayList<Future<Integer>> futures = new ArrayList<>();
		for(int i = 0; i < NUM_GAMES; i++) {
			Callable<Integer> gameParallelism = new gameParallelism(weights, result);
			System.out.println("Threads "+ Thread.currentThread() +" is executing");
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
		System.out.println("All threads are done!");
		System.out.println("Individual with weights "+ Arrays.toString(weights) +" gets "+result);
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
	  
	// reduce the population size after half of the total generations
	// population size decreases exponentially but maintains threshold of 10
  	public static int evolvePopSize(int genCount, int size){
		System.out.println("evolvePopSize(" + genCount + ", " + size + ")");
		boolean decrease = true;
		int newSize = size;
		if (genCount >= NUM_GENS / 2 && decrease) {
			newSize = (int) (size * Math.exp(-1.0 / 5));
			if (newSize < 10) {
				newSize = size;
				decrease = false;
			}
        }
		return newSize;
  	}

	// uses the genetic algorithm and returns the best weights
	public static double[] evolveWeights() {
		int POP_SIZE=100; // the size of the population
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
			int threads = Runtime.getRuntime().availableProcessors();
			ExecutorService executor = Executors.newFixedThreadPool(threads);
			for (int j = 0; j < POP_SIZE; j++) {
				System.out.println("Individual "+ Arrays.toString(population[j].weights) +" is executing");
				Runnable individual = new evolveParallelism(population,j,POP_SIZE);
				executor.execute(individual);
			}
			executor.shutdown();

			POP_SIZE = evolvePopSize(i,POP_SIZE);
			// generate all the children for this generation
			Individual[] allChildren = new Individual[REPLACEMENT_SIZE];
			int childIndex = 0;
			// TODO: may be able to do this on a constant number of threads, then discard extras
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
			avGameScore = Population.getAvGameScore(population);
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

	//the final keyword is used in several contexts to define an entity that can only be assigned once.
	public double findFitness(final int[][] playField, final int[] playTop,double[] tempWgts){
		int maxRow = playField.length;
		int maxCol = playField[0].length;
		// features
		double landingHeight = 0; // Done
		double rowsCleared = 0; // Done
		double rowTransitions = 0; // Done
		double columnTransitions = 0; // Done
		double numHoles = 0; // Done
		double wellSums = 0;
		int moveNumber = -1;

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
				for (int k = j -1; k >=0; k--) {
					if(playField[k][0] == 0) wellSums++;
					else break;
				}
			}
			if(playField[j][maxCol-1] == 0 && playField[j][maxCol-2] != 0) {
				wellSums++;
				for (int k = j -1; k >=0; k--) {
					if(playField[k][maxCol-1] == 0) wellSums++;
					else break;
				}
			}
		}
		return landingHeight*tempWgts[0] + rowsCleared*tempWgts[1]+ rowTransitions*tempWgts[2] +
				columnTransitions*tempWgts[3] + numHoles*tempWgts[4] + wellSums*tempWgts[5];
	}

	public static int playGame(double[] weights) {
		State s = new State();
		new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();
		while(!s.hasLost()) {
			s.makeMove(p.pickMove(weights, s, s.legalMoves()));   //make this optimal move
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

	public int pickMove(double[] weights, State s, int[][] legalMoves) {
		//Variable declaration
		double maxScore = -9999;
		int optimalMove = -9999;
		int oldTop[] = s.getTop();
		int oldField[][] = s.getField();
		for(int moveCount = 0; moveCount < legalMoves.length; moveCount++) {
			int orient = legalMoves[moveCount][0];
			int slot = legalMoves[moveCount][1];
			PlayOnThisBoard playboard = new PlayOnThisBoard(oldField,oldTop);
			// int[][] playField = copyField(oldField);
			// int[] playTop = Arrays.copyOf(oldTop, oldTop.length);
			//do this moving on the copied board
			if(playboard.playMove(s, orient, slot)){
				double tempScore = findFitness(playboard.getPlayField(), playboard.getPlayTop(), weights);
				if(Math.abs(tempScore - maxScore) < 0.000000001){
					//whenever the score is similar,random check update or not
					if(Math.random() > 0.5)
						optimalMove = moveCount;
				}
				else if(tempScore > maxScore){
					//if significantly improved,update
					optimalMove = moveCount;
					maxScore = tempScore;
				}
			}
		}
		if (optimalMove == -9999) {return 0;}
		return optimalMove;
	}

	// TESTED
	public static void plotData(String name, String x, String y, double[] data) {
		System.out.println("Graph for " + name + ":");
		for (double aData : data) {
			System.out.print(aData + " ");
		}
		double[] xa = new double[data.length]; xa[0] = 1;
		for(int i = 1; i < xa.length; i++){
			xa[i] = i+1;
		}
		MatlabChart fig = new MatlabChart();

		fig.plot(xa, data, "-r", 2.0f, "data");
		fig.RenderPlot();
		fig.title(name);
		fig.xlim(0, 100);
		fig.ylim(0, 300);
		fig.xlabel(x);
		fig.ylabel(y);
		fig.grid("on","on");
		fig.legend("northeast");
		fig.font("Helvetica",15);
		fig.saveas(name+".jpeg",640,480);
	}

	public static void runTests() {
		Individual[] pop = Population.initializeRandomPopulation(5);
		pop[0].gameScore = 3;
		pop[1].gameScore = 6;
		pop[2].gameScore = 2;
		pop[3].gameScore = 4;
		Arrays.sort(pop);
		System.out.println("ave = " + Population.getAvGameScore(pop));
		// Population.printPop(pop);

		Individual[] children = pop[0].cross(pop[1], 1.0);
		//Population.printPop(children);

		children[0].mutate(1.0);
		children[0].printInd();

		// double[] data = {100.0, 50.0, 20.0, 80.0};
		// plotData("test", "t", "val", data);

		int currSize = 100;
		for (int i = 0; i < 50; i++) {
			int nextSize = evolvePopSize(i, currSize);
			currSize = nextSize;
			System.out.println("currSize is " + currSize);
		}
	}

	public static void main(String[] args) {
		double[] foundWeights = {-7.25,3.87,-7.25,-7.25,-7.25,-7.25};
		if (args[0].equals("--evolve")) {
			double[] weights = evolveWeights();
			System.out.println("Evolved weights are" + Arrays.toString(weights));
			plotData("scores vs gen", "gen", "score", scores);
			plotData("crossRates vs gen", "gen", "score", crossRates);
			plotData("mutationRates vs gen", "gen", "score", mutationRates);
		}
		else if (args[0].equals("--play")) {
			playGame(foundWeights);
		}
		else if (args[0].equals("--test")){
			runTests();
		}
		else {
			System.out.println("To evolve the weights, use '--evolve'.");
			System.out.println("To play the game using our weights, use '--play'.");
			System.out.println("To run tests, use '--test'.");
		}
	}
}

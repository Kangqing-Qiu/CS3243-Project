import java.util.Arrays;

public class PlayGame {
	public class PlayOnThisBoard{
		private int[][] playField;
		private int[] playTop;
		private PlayOnThisBoard(int[][] oldField,int[] oldTop){
			playTop = Arrays.copyOf(oldTop, oldTop.length);
			playField = new int[oldField.length][];
       		for(int i = 0; i < oldField.length; i++){
            	playField[i] = Arrays.copyOf(oldField[i], oldField[i].length);
        	}
		}

		public int[][] getplayField(){
			return playField;
		}

		public int[] getplayTop(){
			return playTop;
		}

	    public Boolean playMove(State s,int orient,int slot) {
	        // TODO: clarify variables
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
    // similar to makemove in State.java, since we cannot do this on the original board
    //the final keyword is used in several contexts to define an entity that can only be assigned once.
    public double findFitness(final int[][] playField, final int[] playTop,double[] tempWgts){
        // TODO: clarify variables
        int maxRow = playField.length;
        int maxCol = playField[0].length;
        //temp test features
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

    public int pickMove(double[] weights, State s, int[][] legalMoves) {
        // TODO: clarify variables
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
                double tempScore = findFitness(playboard.getplayField(), playboard.getplayTop(), weights);
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
}

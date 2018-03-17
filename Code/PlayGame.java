import java.util.*;
import java.io.*;
import java.util.Random;

public class PlayerSkeleton {

    public static int[][] copyField(int[][] oldField){
        int[][] copiedField = new int[oldField.length][];
        int length = oldField[0].length;
        for(int i = 0; i < oldField.length; i++){
            copiedField[i] = Arrays.copyOf(oldField[i], length);
        }
        return copiedField;
    }

    // similar to makemove in State.java, since we cannot do this on the original board
    //the final keyword is used in several contexts to define an entity that can only be assigned once.
    public Boolean playMove(final State s, final int orient,final int slot,int[][] playField,final int[] playTop) {
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
                playField[h][i+slot] = turn;
            }
        }

        //adjust top
        for(c = 0; c < pWidth[nextPiece][orient]; c++) {
            playTop[slot+c] = height+pTop[nextPiece][orient][c];
        }
        return true;
    }

    public double findFitness(final int[][] playField, final int[] playTop,double[] tempWgts){
        int maxRow = playField.length;
        int maxCol = playField[0].length;
        //temp test features
        double landingHeight = 0;
        double rowsCleared = 0;
        for(int i = 0; i<maxCol; i++) {
            for (int j  = newTop[i]-1; j >=0; j--) {
                if(newField[j][i] == 0) numHoles++;
            }
            if(newField[Math.max(newTop[i]-1, 0)][i] > moveNumber) {
                moveNumber = newField[Math.max(newTop[i]-1, 0)][i];
                
                landingHeight = newTop[i];
            }
        }
        for(int i = 0; i<maxRow; i++) {
            boolean lastCell = false;
            boolean currentCell = false;
            int rowIsClear = 1;
            for (int j = 0; j<maxCol; j++) {
                currentCell = false;
                if(newField[i][j] == 0) {
                    rowIsClear = 0;
                    currentCell = true;
                }
                
                if(lastCell != currentCell) {
                    rowTransitions++;
                }
                lastCell = currentCell;
            }
            rowsCleared += rowIsClear;
            if(currentCell) rowTransitions++;
        }
        return landingHeight*tempWgts[0] + rowsCleared*tempWgts[1];
        
    }

    public int pickMove(State s, int[][] legalMoves) {
        //Variable declaration
        tempWeights =  new double[] {-7.25,3.87};
        double maxScore = -9999;
        int optimalMove = -9999;
        int oldTop[] = s.getTop();
        int oldField[][] = s.getField();
        for(int moveCount = 0; moveCount < legalMoves.length; moveCount++) {
            int orient = legalMoves[i][0];
            int slot = legalMoves[moveCount][1];
            int[][] playField = copyField(oldField);
            int[] playTop = Arrays.copyOf(oldTop, oldTop.length);
            //do this moving on the copied board
            if(playMove(s, orient, slot, playField, playTop)){
                double tempScore = findFitness(playField, playTop, tempWeights);
                if(Math.abs(score - highestScore) < 0.000000001){
                    //whenever the score is similar,random check update or not
                    if(Math.random() > 0.5)
                        optimalMove = moveCount;
                }
                else if(score > highestScore){
                    //if significantly improved,update
                    optimalMove = moveCount;
                    maxScore = tempScore;
                }
            }
        }
        if (optimalMove == -9999) {return 0;}
        return optimalMove;
    }
    
    public static void main(String[] args) {
        State s = new State();
        new TFrame(s);
        PlayerSkeleton p = new PlayerSkeleton();
        while(!s.hasLost()) {
            s.makeMove(p.pickMove(s,s.legalMoves()));   //make this optimal move
            s.draw();
            s.drawNext(0,0);
            try {
                Thread.sleep(300);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        System.out.println("You have completed "+s.getRowsCleared()+" rows.");
    }
}
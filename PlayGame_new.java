import java.util.*;
import java.io.*;
import java.util.Random;

public class PlayerSkeleton {

    public static int[][] FieldCopy(int[][] OriField){
        int[][] CopiedField = new int[OriField.length][];
        for(int FieldCount = 0; FieldCount < OriField.length; FieldCount ++){
            CopiedField[FieldCount] = Arrays.copyOf(OriField[FieldCount], OriField[FieldCount].length);
        }
        return CopiedField;
    }

    public Boolean PlayMove(final int orient,final int slot,int[][] PlayField,final int[] PlayTop,final State Nows) {
        //the final keyword is used in several contexts to define an entity that can only be assigned once.
        //imitate the makemove operation,can't do this on the original board
        //similar to makemove in State.java
        int pWidth[][] = Nows.getpWidth();
        int pHeight[][] = Nows.getpHeight();
        int pTop[][][] = Nows.getpTop();
        int pBottom[][][] = Nows.getpBottom();
        int nextPiece = Nows.getNextPiece();
        int turnNumber = Nows.getTurnNumber();

        turnNumber++;
        //height if the first column makes contact
        int height = PlayTop[slot]-pBottom[nextPiece][orient][0];
        //for each column beyond the first in the piece
        for(int count1 = 1; count1 < pWidth[nextPiece][orient]; count1 ++) {
            height = Math.max(height,PlayTop[slot + count1]- pBottom[nextPiece][orient][count1]);
        }
        //check if game ended
        if(height+pHeight[nextPiece][orient] >= State.ROWS) {
            return false;
        }
        //for each column in the piece - fill in the appropriate blocks
        for(int i = 0; i < pWidth[nextPiece][orient]; i++) {
            //from bottom to top of brick
            for(int h = height+pBottom[nextPiece][orient][i]; h < height+pTop[nextPiece][orient][i]; h++) {
                PlayField[h][i+slot] = turn;
            }
        }
        //adjust top
        for(count1 = 0; count1 < pWidth[nextPiece][orient]; count1++) {
            PlayTop[slot+count1] = height+pTop[nextPiece][orient][count1];
        }
        return true;
    }

    public double FindFitness(final int[][] PlayField, final int[] PlayTop,double[] TempWgts){
        int maxRow = PlayField.length;
        int maxCol = PlayField[0].length;
        //temp test features
        double landingHeight = 0; // Done
        double rowsCleared = 0; // Done
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
            rowsCleared+=rowIsClear;
            if(currentCell) rowTransitions++;
        }
        return landingHeight*TempWgts[0] + rowsCleared*TempWgts[1];
        
    }

    //implement this function to have a working system
    public int pickMove(State s, int[][] legalMoves) {
        //变量声明
        TempWeights =  new double[] {-7.25,3.87};
        double MaxScore = -9999;
        int Oritop[] = s.getTop();
        int Orifield[][] = s.getField();
        for(int MoveCount = 0; MoveCount < legalMoves.length; MoveCount ++) {
            int orient = legalMoves[MoveCount][0];
            int slot = legalMoves[MoveCount][1];
            int[][] PlayField = FieldCopy(Orifield);
            int[] PlayTop = Arrays.copyOf(Oritop, Oritop.length);
            //if it is a legal move
            if(PlayMove(orient, slot, PlayField, PlayTop, s)){
                double TempScore = FindFitness(PlayField, PlayTop, TempWeights);
                //是否考虑相似的分数，保留多个moves？？？？
                //有三种形式的makemove，所以这个函数应该返回什么？
                //这个possiblemove记录的具体含义是？？
                /*
                if(Math.abs(score - highestScore) < 0.000000001){

                    possibleMoves.add(i);
                }
                else if(score > highestScore){
                    possibleMoves.clear();
                    possibleMoves.add(i);
                    highestScore = score;
                }
                */
            }
        }
        /*
        return possibleMoves.size() == 0 ? 0 : possibleMoves.get(randnum.nextInt(possibleMoves.size()));
        */
        return 0;
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
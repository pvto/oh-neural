package oh.neural;

/**
 *
 * @author Paavo Toivanen https://github.com/pvto
 */
public class Prob {

    
    public static void fillFromU(double[] A, double lowInclusive, double highExclusive)
    {
        fillFromU(A, lowInclusive, highExclusive, 0, A.length - 1);
    }
    
    
    public static void fillFromU(double[] A, double lowInclusive, double highExclusive, int startIndex, int endIndex)
    {
        for(int i = startIndex; i <= endIndex; i++)
            A[i] = Math.random() * (highExclusive - lowInclusive) + lowInclusive;
    }

}

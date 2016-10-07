package oh.neural;

/**
 *
 * @author Paavo Toivanen https://github.com/pvto
 */
public class Ints {

    public static final int[] fill(int size, int x)
    {
        int[] ret = new int[size];
        for (int i = 0; i < ret.length; i++)
            ret[i] = x;
        return ret;
    }
}

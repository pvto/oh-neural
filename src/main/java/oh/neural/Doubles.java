package oh.neural;

import java.util.Arrays;

/**
 *
 * @author Paavo Toivanen https://github.com/pvto
 */
public final class Doubles {  private Doubles(){}
    
    public static final double[] column(double[][] AA, int column)
    {
        double[] ret = new double[AA.length];
        for (int i = 0; i < ret.length; i++)
            ret[i] = AA[i][column];
        return ret;
    }
    
    public static final double sum(double[] A)
    {
        double sum = 0.0;
        for(int i = 0; i < A.length; i++)
            sum += A[i];
        return sum;
    }
    
    public static final double[] fill(int size, double x)
    {
        double[] ret = new double[size];
        for (int i = 0; i < ret.length; i++)
            ret[i] = x;
        return ret;
    }

    public static final double[][] fill(int size, double[] A)
    {
        double[][] ret = new double[size][];
        for (int i = 0; i < ret.length; i++)
            ret[i] = Arrays.copyOf(A, A.length);
        return ret;
    }
    
    public static final double[] sub(double[] A, double[] B)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = A[i] - B[i];
        return ret;
    }
    
    public static final double[] add(double[] A, double[] B)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = A[i] + B[i];
        return ret;
    }

    public static final double[] mul(double[] A, double[] B)
    {
        return dot(A, B);
    }
    
    public static final double[] dot(double[] A, double[] B)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = A[i] * B[i];
        return ret;
    }

    public static final double[] div(double[] A, double[] B)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = A[i] / B[i];
        return ret;
    }

    public static final double[] sub(double[] A, double b)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = A[i] - b;
        return ret;
    }
    
    public static final double[] add(double[] A, double b)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = A[i] + b;
        return ret;
    }
    
    public static final double[] mul(double[] A, double b)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = A[i] * b;
        return ret;
    }
    
    public static final double[] div(double[] A, double b)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = A[i] / b;
        return ret;
    }
    
    public static final double[] abs(double[] A)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = Math.abs(A[i]);
        return ret;
    }
        
    public static final double[] counter(double[] A)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = 1.0 / A[i];
        return ret;
    }

    public static final double[] sqr(double[] A)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = A[i] * A[i];
        return ret;
    }
    
    public static final double[] sqrt(double[] A)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = Math.sqrt(A[i]);
        return ret;
    }
    
    public static final double[] pow(double[] A, double exp)
    {
        double[] ret = new double[A.length];
        for(int i = 0; i < A.length; i++)
            ret[i] = Math.pow(A[i], exp);
        return ret;
    }
}

package oh.neural;

import oh.neural.Fn.Composing.ComposingDd;

/** Function interfaces.
 *
 * About naming.
 *
 * First character in a name is written in capitals and signifies return type.
 * The chars that follow are written in lower case and signify argument types
 * from left to right.
 *
 * D,d double;
 * F,f float;
 * I,i int;
 * L,l long;
 * C,c char;
 * S,s String;
 * O,o Object;
 *
 * About naming partial functions (Fn.Partials).
 *
 * Characters signifying the return type and the argument types are same as before.
 *
 * Character 'X' is used as a marker BEFORE a free variable.
 *
 * Thus partialDXdd(Ddd f, double p) returns a Dd(x) that
 * computes and returns f(x, p).
 *
 *
 * @author Paavo Toivanen https://github.com/pvto
 */
public interface Fn {


    interface D extends Fn          { double f(); }
    interface Dd extends Fn         { double f(double a); }
    interface Ddd extends Fn        { double f(double a, double b); }
    interface Dddd extends Fn       { double f(double a, double b, double c); }

    interface Ddi extends Fn        { double f(double a, int b); }

    public interface Composing {

        Fn underlyingFunction(int index);


        public static Fn underlyingFunction(Fn function, int index)
        {
            return Composing.class.isAssignableFrom(function.getClass()) ?
                    ((Composing)function).underlyingFunction(index)
                    : null ;
        }


        interface ComposingD extends D, Composing {}
        interface ComposingDd extends Dd, Composing {}
        interface ComposingDdd extends Ddd, Composing {}
        interface ComposingDddd extends Dddd, Composing {}
    }

    public static final class Partials
    {

        public static final ComposingDd partialDXdd(final Ddd base, final double param2)
        {
            ComposingDd f = new ComposingDd() {
                @Override
                public double f(double a)
                {
                    return base.f(a, param2);
                }

                @Override
                public Fn underlyingFunction(int index)
                {
                    if (0 != index)
                        throw new IllegalArgumentException(this.getClass().getSimpleName() + " is composed of exactly one underlying function (use index=0)");
                    return base;
                }
            };
            return f;
        }

        public static final ComposingDd partialDXdi(final Ddi base, final int param2)
        {
            ComposingDd f = new ComposingDd() {
                @Override
                public double f(double a)
                {
                    return base.f(a, param2);
                }

                @Override
                public Fn underlyingFunction(int index)
                {
                    if (0 != index)
                        throw new IllegalArgumentException("partialDxdi is composed of exactly one underlying function (use index=0)");
                    return base;
                }
            };
            return f;
        }
    }

    public static final class Transfer {

        public static final Ddd linear = new Linear();
        public static final Ddd linearD1 = new LinearD1();
        public static final Ddi staircase = new Staircase();
        public static final Ddi staircaseD1 = new StaircaseD1();
        public static final Ddd sigmoid = new Sigmoid();
        public static final Ddd sigmoidD1 = new SigmoidD1();
        public static final Ddd tanh = new Tanh();
        public static final Ddd tanhD1 = new TanhD1();
        public static final Dd  softsign = new Softsign();
        public static final Dd  softsignD1 = new SoftsignD1();
        public static final Ddd gaussian = new Gaussian();
        public static final Ddd gaussianD1 = new GaussianD1();

        public static final Dd linear(double slope)         { return Fn.Partials.partialDXdd(linear, slope); }
        public static final Dd linearD1(double slope)       { return Fn.Partials.partialDXdd(linearD1, slope); }
        public static final Dd staircase(int steps)         { return Fn.Partials.partialDXdi(staircase, steps); }
        public static final Dd staircaseD1(int steps)       { return Fn.Partials.partialDXdi(staircaseD1, steps); }
        public static final Dd sigmoid(double steepness)    { return Fn.Partials.partialDXdd(sigmoid, steepness); }
        public static final Dd sigmoidD1(double steepness)  { return Fn.Partials.partialDXdd(sigmoidD1, steepness); }
        public static final Dd tanh(double steepness)       { return Fn.Partials.partialDXdd(tanh, steepness); }
        public static final Dd tanhD1(double steepness)     { return Fn.Partials.partialDXdd(tanhD1, steepness); }
        public static final Dd softsign()                   { return softsign; }
        public static final Dd softsignD1()                 { return softsignD1; }
        public static final Dd gaussian(double sigma)       { return Fn.Partials.partialDXdd(gaussian, sigma); }
        public static final Dd gaussianD1(double sigma)     { return Fn.Partials.partialDXdd(gaussianD1, sigma); }



        public static final class Linear implements Ddd
        {
            public double f(double x, double steepness)
            {
                return x * steepness;
            }
        }

        public static final class LinearD1 implements Ddd
        {
            public double f(double x, double steepness)
            {
                return steepness;
            }
        }

        public static final class Staircase implements Ddi
        {
            double[] levels;
            double[] d1Levels;
            double[] thresholds;

            public void initLevels(int steps)
            {
                if (levels == null) {
                    Dd unlinear = tanh(3d);
                    Dd unlinearD1 = tanhD1(3d);
                    levels = new double[steps];
                    d1Levels = new double[steps];
                    thresholds = new double[steps];
                    double step = 1d / (double)(steps + 1);
                    double d = -1d + 0.5d * step;
                    for(int i = 0; i < steps; i++) {
                        thresholds[i] = d;
                        levels[i] = unlinear.f(d);
                        d1Levels[i] = unlinearD1.f(d);
                        d = d + step;
                    }
                }
            }

            public double f(double x, int steps)
            {
                initLevels(steps);
                for(int i = 0; i < levels.length; i++)
                {
                    if (thresholds[i] >= x)
                    {
                        return levels[i];
                    }
                }
                return levels[levels.length - 1];
            }
        }

        public static final class StaircaseD1 implements Ddi
        {
            Staircase staircase;

            public double f(double x, int steps)
            {
                if (staircase == null)
                {
                    staircase = new Staircase();
                    staircase.initLevels(steps);
                }
                for(int i = 0; i < staircase.d1Levels.length; i++)
                {
                    if (staircase.thresholds[i] >= x)
                    {
                        return staircase.d1Levels[i];
                    }
                }
                return staircase.d1Levels[staircase.d1Levels.length - 1];
            }
        }

        public static final class Sigmoid implements Ddd
        {
            public double f(double x, double steepness)
            {
                if (x > 100)
                    return 1.0;
                if (x < -100)
                    return 0f;
                double activation =
                        1.0
                        /
                        (1.0 + Math.exp(steepness * x));
                return activation;
            }
        }

        public static final class SigmoidD1 implements Ddd
        {
            public double f(double x, double steepness)
            {
                return steepness * x * (1.0 - x)
                    + 0.1; // fix for the "flat spot problem"
            }
        }

        public static class Tanh implements Ddd
        {
            @Override
            public double f(double x, double steepness)
            {
                if (x > 100.0) {
                    return 1.0;
                } else if (x < -100.0) {
                    return -1.0;
                }

                float E_x = (float) Math.exp(steepness * x);
                return (E_x - 1f) / (E_x + 1f);
            }
        }

        public static final class TanhD1 extends Tanh implements Ddd
        {
            @Override
            public double f(double x, double steepness)
            {
                double out = super.f(x, steepness);
                return (1.0 - out * out);
            }
        }

        public static final class Softsign implements Dd
        {
            @Override
            public double f(double x)
            {
                if (x == 0.0)
                    return 0.0;
                double absval = x > 0.0 ? x : -x;
                return x / (1.0 + absval);
            }
        }

        public static final class SoftsignD1 implements Dd
        {
            @Override
            final public double f(double x)
            {
                return 1.0 /
                        (1.0 + 2.0*x + x*x);

            }
        }

        public static class Gaussian implements Ddd
        {
            @Override
            public double f(double x, double sigma)
            {
                return Math.exp(
                        - (x * x)
                        /
                        (2 * sigma * sigma)
                );
            }
        }

        public static final class GaussianD1 extends Gaussian
        {
            @Override
            public double f(double x, double sigma)
            {
                double val = super.f(x, sigma);
                return val * ( -x / (sigma * sigma) );
            }
        }

    }

}

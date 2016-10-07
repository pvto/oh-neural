
package oh.neural;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import oh.neural.FeedForwardNetwork.NodeLayer;
import org.junit.Test;

public class FeedForwardNetworkTest {

    @Test
    public void testFeedForward() {
        NodeLayer in = new NodeLayer(1);
        NodeLayer a = new NodeLayer(15);
        NodeLayer b = new NodeLayer(6);
        NodeLayer c = new NodeLayer(6);
        NodeLayer out = new NodeLayer(1);
        {
            double steepness = 3.0;
            a.transferFunction = Fn.Transfer.staircase(8); //sigmoid(steepness);
            a.deltaFunction = Fn.Transfer.staircaseD1(8); //sigmoidD1(steepness);
        }
        {
            double steepness = 1.0;
            b.transferFunction = Fn.Transfer.sigmoid(steepness);
            b.deltaFunction = Fn.Transfer.sigmoidD1(steepness);
        }
        {
            double steepness = 3.0;
            c.transferFunction = Fn.Transfer.sigmoid(steepness);
            c.deltaFunction = Fn.Transfer.sigmoidD1(steepness);
        }
        {
            double slope = 1.0;
            out.transferFunction = Fn.Transfer.linear(slope);
            out.deltaFunction = Fn.Transfer.linearD1(slope);
//            out.transmissionFunction = Fn.Squashings.sigmoid(steepness);
//            out.deltaFunction = Fn.Squashings.sigmoidD1(steepness);
//            out.modulationFunction = new Ddd() {
//                @Override
//                public double f(double ind, double activation) {
//                    return activation * normalize / sum * ind;
//                }
//            };
        }
        List<NodeLayer> system = new ArrayList<>();
        a.addFeedingLayer(in, system);
        out.addFeedingLayer(a, system);

//        a.addFeedingLayer(a, system);
//        b.addFeedingLayer(in, system);
//        out.addFeedingLayer(b, system);
//
//        c.addFeedingLayer(in, system);
//        out.addFeedingLayer(c, system);

//        out.addFeedingLayer(in, system);
        int[] targetValueOffsets = Ints.fill(system.size(), -1);
        {
            targetValueOffsets[system.indexOf(out)] = 1;
        }
        double[][] lrc = new double[system.size()][];
        {
            for(int i = 0; i < lrc.length; i++)
                lrc[i] = Doubles.fill(system.get(i).nodeValues.length, 0.001);
        }


        double[][] data = new double[300][];
        for(int i = 0; i < data.length; i++)
        {
            double x = Math.random();
            double targ = 1 - (x * x + Math.sin(x*10)*0.2);
            double[] sample = new double[]{
                x,
                targ,
                0 // extra column for observed output...
            };
            data[i] = sample;
        }

        int split = data.length * 7 / 10;
        int maxEpochs = 8000;
        double errorMin = Double.MAX_VALUE;
        double errorAgg = 0;
soOut:  for(int epoch = 0; epoch < maxEpochs; epoch++)
        {
            for(int s = 0; s < split; s++)
            {
                if (epoch == 0 && s == 1)
                {
                    System.out.println("Item: " + Arrays.toString(data[s]));
                    for(NodeLayer nodeLayer : system)
                        nodeLayer.print(System.out);
                }
                in.nodeValues = Arrays.copyOfRange(data[s], 0, 1);

                FeedForwardNetwork.feedForward(system);
                FeedForwardNetwork.backpropagate(system, data[s], targetValueOffsets, lrc);

                double[] errors = new double[data.length - split];
                for(int s2 = split; s2 < data.length; s2++)
                {
                        in.nodeValues = Arrays.copyOfRange(data[s2], 0, in.size());
                        double[] D = Arrays.copyOfRange(data[s2], targetValueOffsets[system.indexOf(out)], targetValueOffsets[system.indexOf(out)]+out.size());
                        a.feedForward();
                        out.feedForward();
                        double[] errorTermsOut = Doubles.sub(out.nodeValues, D);
                        errors[s2 - split] = Doubles.sum(out.nodeValues) - Doubles.sum(D);
                }
                double errorNow = (Doubles.sum(Doubles.abs(errors)) / errors.length);
                errorMin = Math.min(errorNow, errorMin);
                if (epoch % 10 == 0 && s == 0)
                {
                    System.out.println("avg error ("+epoch+" epocs) = "
                            + errorNow );
                }
                errorAgg = (errorAgg * 3.0 + errorNow) / 4.0;
                if (epoch > 10 && errorNow > errorAgg && errorNow > errorMin + 0.05)
                {
                    break soOut;
                }
            }

        }

        try {
            BufferedWriter fout = new BufferedWriter(new FileWriter("out.csv"));
            fout.write("X,T,O,res\n");
            for (int s = 0; s < data.length; s++)
            {
                in.nodeValues = Arrays.copyOfRange(data[s], 0, 1);
                FeedForwardNetwork.feedForward(system);
                FeedForwardNetwork.backpropagate(system, data[s], targetValueOffsets, lrc);
                System.arraycopy(out.nodeValues, 0, data[s], data[s].length - out.nodeValues.length, out.nodeValues.length);
                double[] ww = data[s];
                for(int i = 0; i < ww.length; i++)
                {
                    fout.write(String.format("%.3f,", ww[i]));
                }
                fout.write("\n");

            }
            fout.flush();
            fout.close();
        } catch(Exception e){
            e.printStackTrace();
        }

    }

}

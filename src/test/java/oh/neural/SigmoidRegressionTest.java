package oh.neural;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Ignore;
import org.junit.Test;

/**
 *
 * @author Paavo Toivanen https://github.com/pvto
 */
public class SigmoidRegressionTest {

    @Ignore
    @Test
    public void testMultiInput() {
        FeedForwardNetwork.NodeLayer in = new FeedForwardNetwork.NodeLayer(3);
        FeedForwardNetwork.NodeLayer a = new FeedForwardNetwork.NodeLayer(25);
        FeedForwardNetwork.NodeLayer out = new FeedForwardNetwork.NodeLayer(1);
        { 
            double steepness = 3.0; 
            a.transferFunction = Fn.Transfer.sigmoid(steepness); 
            a.deltaFunction = Fn.Transfer.sigmoidD1(steepness);
        }
        { 
            double steepness = 3.0; 
            out.transferFunction = Fn.Transfer.sigmoid(steepness);
            out.deltaFunction = Fn.Transfer.sigmoid(steepness);
        }
        List<FeedForwardNetwork.NodeLayer> system = new ArrayList<>();
        a.addFeedingLayer(in, system);
        out.addFeedingLayer(a, system);

        int[] targetValueOffsets = Ints.fill(system.size(), -1);
        {
            targetValueOffsets[system.indexOf(out)] = 1;
        }
        double[][] lrc = new double[system.size()][];
        {
            for(int i = 0; i < lrc.length; i++)
                lrc[i] = Doubles.fill(system.get(i).nodeValues.length, 0.01);
        }
        
        
        double[][] data = new double[300][];
        for(int i = 0; i < data.length; i++)
        {
            double x = Math.random();
            double targ = 1 - (x * x + Math.sin(x*10)*0.2);
            double[] sample = new double[]{
                x<0.2?x:0,
                x>=0.2&&x<0.5?x:0,
                x>=0.5?x:0,
                targ,
                0 // extra column for observed output...
            };
            data[i] = sample;
        }
                
        int split = data.length * 7 / 10;
        int maxEpochs = 32000;
        double errorMin = Double.MAX_VALUE;
        double errorAgg = 0;
soOut:  for(int epoch = 0; epoch < maxEpochs; epoch++)
        {
            for(int s = 0; s < split; s++)
            {
                if (epoch == 0 && s == 1)
                {
                    System.out.println("Item: " + Arrays.toString(data[s]));
                    for(FeedForwardNetwork.NodeLayer nodeLayer : system)
                        nodeLayer.print(System.out);
                }
                in.nodeValues = Arrays.copyOfRange(data[s], 0, in.size());
                
                FeedForwardNetwork.feedForward(system);
                FeedForwardNetwork.backpropagate(system, data[s], targetValueOffsets, lrc);
                
            }
            
            double[] errors = new double[data.length - split];
            for(int s2 = split; s2 < data.length; s2++)
            {
                    in.nodeValues = Arrays.copyOfRange(data[s2], 0, in.size());
                    a.feedForward();
                    out.feedForward();
                    double[] D = Arrays.copyOfRange(data[s2], targetValueOffsets[system.indexOf(out)], targetValueOffsets[system.indexOf(out)]+out.size());
                    errors[s2 - split] = Doubles.sum(out.nodeValues) - Doubles.sum(D);
            }
            double errorNow = (Doubles.sum(Doubles.abs(errors)) / errors.length);
            errorMin = Math.min(errorNow, errorMin);
            if (epoch % 10 == 0)
            {
                System.out.println("avg error ("+epoch+" epocs) = " 
                        + errorNow );
            }
            errorAgg = (errorAgg * 3.0 + errorNow) / 4.0;
            if (epoch > 40 && errorNow > errorAgg && errorNow > errorMin + 0.05)
            {
                break soOut;
            }
            
        }

        try {
            BufferedWriter fout = new BufferedWriter(new FileWriter("out2.csv"));
            fout.write("X1,X2,X3, T,O\n");
            for (int s = 0; s < data.length; s++)
            {
                in.nodeValues = Arrays.copyOfRange(data[s], 0, in.size());
                FeedForwardNetwork.feedForward(system);
                FeedForwardNetwork.backpropagate(system, data[s], targetValueOffsets, lrc);
                System.arraycopy(out.nodeValues, 0, data[s], data[s].length - out.nodeValues.length, out.nodeValues.length);
                double[] ww = data[s];
                for(int i = 0; i < ww.length; i++)
                {
                    if (i > 0)
                        fout.write(",");
                    fout.write(String.format("%.3f", ww[i]));
                }
                fout.write("\n");
                
            }
            fout.flush();
            fout.close();
        } catch(Exception e){
            e.printStackTrace();
        }
        
    }

    
    
    
    
    @Test
    public void testMultiOutput() {
        FeedForwardNetwork.NodeLayer in = new FeedForwardNetwork.NodeLayer(1);
        FeedForwardNetwork.NodeLayer a = new FeedForwardNetwork.NodeLayer(12);
        FeedForwardNetwork.NodeLayer out = new FeedForwardNetwork.NodeLayer(3);
        { 
            double steepness = 3.0; 
            a.transferFunction = Fn.Transfer.sigmoid(steepness); 
            a.deltaFunction = Fn.Transfer.sigmoidD1(steepness);
        }
        { 
            double steepness = 3.0; 
            out.transferFunction = Fn.Transfer.sigmoid(steepness);
            out.deltaFunction = Fn.Transfer.sigmoid(steepness);
        }
        List<FeedForwardNetwork.NodeLayer> system = new ArrayList<>();
        a.addFeedingLayer(in, system);
        out.addFeedingLayer(a, system);

        int[] targetValueOffsets = Ints.fill(system.size(), -1);
        {
            targetValueOffsets[system.indexOf(out)] = 1;
        }
        double[][] lrc = new double[system.size()][];
        {
            for(int i = 0; i < lrc.length; i++)
                lrc[i] = Doubles.fill(system.get(i).nodeValues.length, 0.01);
        }
        
        
        double[][] data = new double[300][];
        for(int i = 0; i < data.length; i++)
        {
            double x = Math.random();
            double targ = 1 - (x * x + Math.sin(x*10)*0.2);
            double[] sample = new double[]{
                x,
                x<0.333?targ:0,
                x>=0.333&&x<0.666?targ:0,
                x>=0.666?targ:0,
                0,0,0 // extra columns for observed output...
            };
            data[i] = sample;
        }
                
        int split = data.length * 7 / 10;
        int maxEpochs = 100;
        double errorMin = Double.MAX_VALUE;
        double errorAgg = 0;
soOut:  for(int epoch = 0; epoch < maxEpochs; epoch++)
        {
            for(int s = 0; s < split; s++)
            {
                if (epoch == 0 && s == 1)
                {
                    System.out.println("Item: " + Arrays.toString(data[s]));
                    for(FeedForwardNetwork.NodeLayer nodeLayer : system)
                        nodeLayer.print(System.out);
                }
                in.nodeValues = Arrays.copyOfRange(data[s], 0, in.size());
                
                FeedForwardNetwork.feedForward(system);
                FeedForwardNetwork.backpropagate(system, data[s], targetValueOffsets, lrc);
                
            }
            
            double[] errors = new double[data.length - split];
            for(int s2 = split; s2 < data.length; s2++)
            {
                    in.nodeValues = Arrays.copyOfRange(data[s2], 0, in.size());
                    a.feedForward();
                    out.feedForward();
                    double[] D = Arrays.copyOfRange(data[s2], targetValueOffsets[system.indexOf(out)], targetValueOffsets[system.indexOf(out)]+out.size());
                    errors[s2 - split] = Doubles.sum(out.nodeValues) - Doubles.sum(D);
            }
            double errorNow = (Doubles.sum(Doubles.abs(errors)) / errors.length);
            errorMin = Math.min(errorNow, errorMin);
            if (epoch % 10 == 0)
            {
                System.out.println("avg error ("+epoch+" epocs) = " 
                        + errorNow );
            }
            errorAgg = (errorAgg * 3.0 + errorNow) / 4.0;
            if (epoch > 40 && errorNow > errorAgg && errorNow > errorMin + 0.05)
            {
                break soOut;
            }
            
        }

        try {
            BufferedWriter fout = new BufferedWriter(new FileWriter("multi-out-regr.csv"));
            fout.write("X,Y1,Y2,Y3, O1,O2,O3,O\n");
            for (int s = 0; s < data.length; s++)
            {
                in.nodeValues = Arrays.copyOfRange(data[s], 0, in.size());
                FeedForwardNetwork.feedForward(system);
                FeedForwardNetwork.backpropagate(system, data[s], targetValueOffsets, lrc);
                System.arraycopy(out.nodeValues, 0, data[s], data[s].length - out.nodeValues.length, out.nodeValues.length);
                double[] ww = data[s];
                for(int i = 0; i < ww.length; i++)
                {
                    if (i > 0)
                        fout.write(",");
                    fout.write(String.format("%.3f", ww[i]));
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

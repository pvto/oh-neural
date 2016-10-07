package oh.neural;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import oh.neural.Fn.Dd;
import oh.neural.Fn.Ddd;

/**
 *
 * @author Paavo Toivanen https://github.com/pvto
 */
public class FeedForwardNetwork {

    
    /** Feeds activation from input layer(s) in correct order towards output layer(s) */
    public static void feedForward(List<NodeLayer> system)
    {
        for(NodeLayer layer : system)
            layer.feedForward();
    }
    
    /** Learning, backpropagates error deltas in correct order from output layer(s) back 
     * towards input layer(s)
     * @param system the system of layers
     * @param sample sample, containing supervision target values at indices given by the next parameter
     * @param targetValueOffsets index values for given sample where output layer(s)' target values are to be found. 
     * Their indices match the indices of respective layers in supplied system (list).
     * @param learningRateCoefficients individuated for each neuron
     */
    public static void backpropagate(List<NodeLayer> system, double[] sample, 
            int[] targetValueOffsets,
            double[][] learningRateCoefficients)
    {
        Set<Integer> done = new HashSet<>();
        while(done.size() < system.size() - 1)
            for(int x = 0; x < system.size(); x++)
            {
                NodeLayer xLayer = system.get(x);
                if (xLayer.receivingLayers.size() == 0)
                {
                    double[] targetTermsOut = Arrays.copyOfRange(sample, 
                            targetValueOffsets[x], 
                            targetValueOffsets[x] + xLayer.nodeValues.length
                    );
                    xLayer.errorTerms = Doubles.sub(targetTermsOut, xLayer.nodeValues);
                    done.add(xLayer.hashCode());
                    xLayer.backpropagate(xLayer.errorTerms, learningRateCoefficients[x]);
                }
                else
                {
                    boolean canDo = true;
                    for(NodeLayer receiving : xLayer.receivingLayers)
                    {
                        canDo &= 
                                (done.contains(receiving.hashCode())
                                || system.indexOf(receiving) <= system.indexOf(xLayer)
                                || receiving == xLayer);
                        if (!canDo)
                            break;
                    }
                    if (canDo)
                    {
                        xLayer.computeHiddenLayerErrorTerms();
                        done.add(xLayer.hashCode());
                        xLayer.backpropagate(xLayer.errorTerms, learningRateCoefficients[x]);
                    }
                }
            }
    }
    
    

    public interface InputFunction {

        double f(NodeLayer layer, int nodeIndex); // signature would be more correct as f(double[] inputWeights, double[] inputNeuronValues)... can we sacrifice public double[] NodeLayer.nodeValues ?
        
        public static final InputFunction dot       = new Dot();
        public static final InputFunction diff      = new Diff();
        
        
        public static final class Dot implements InputFunction {
            @Override
            public double f(NodeLayer layer, int nodeIndex)
            {
                double sum = 0.0;
                for (int k = 0; k < layer.feedingLayers.size(); k++)
                {
                    NodeLayer feedingLayer = layer.feedingLayers.get(k);
                    for(int j = 0; j < feedingLayer.nodeValues.length; j++)
                    {
                        sum += feedingLayer.nodeValues[j] * layer.transmissionWeights[k][nodeIndex][j];
                    }
                }
                return sum;
            }
        }
        
        public static final class Diff implements InputFunction {
            @Override
            public double f(NodeLayer layer, int nodeIndex)
            {
                double sum = 0.0;
                for (int k = 0; k < layer.feedingLayers.size(); k++)
                {
                    NodeLayer feedingLayer = layer.feedingLayers.get(k);
                    for(int j = 0; j < feedingLayer.nodeValues.length; j++)
                    {
                        double delta = feedingLayer.nodeValues[j] - layer.transmissionWeights[k][nodeIndex][j];
                        sum += delta * delta;
                    }
                }
                return Math.sqrt(sum);
            }
        }
    }
    
    
    
    
    public static class NodeLayer {
        
        public double[] nodeValues;
        public double[] errorDeltas; // used by backpropagation learning
        public List<NodeLayer> feedingLayers = new ArrayList<>();
        public List<NodeLayer> receivingLayers = new ArrayList<>();
        public double[][][] transmissionWeights;
        
        /** Computes the input that a node receives from feeding nodes and associated feeding weights */
        public InputFunction inputFunction = InputFunction.dot;
        /** Gate function on input, applied first; receives input and current node value; 
         * @returns gate output. */
        public Ddd gateFunction; 
        /** A squashing transmission function on input, like, a sigmoid or tanh function. */
        public Dd transferFunction; 
        /** This is used in error backpropagation; 
         * one should use the first derivative of transmissionFunction */
        public Dd deltaFunction;
        public double[] errorTerms;
        /** Computes a value from node value.
         *  Used with getModulatedActivation().
         * Receives node index in layer and node value.
         *  */ 
        public Ddd modulationFunction;

        public NodeLayer(int size)
        {
            nodeValues = new double[size];
            errorDeltas = new double[size];
            errorTerms = new double[size];
            transmissionWeights = new double[0][][];
        }
        
        public int size() { return nodeValues.length; }
        
        /** Locks a feeding layer to this layer.
         * 
         * Additionally adds this layer, if not already added,
         * and feeding layer, if not already added,
         * to the given system of layers.
         * Feeding layer is placed just before this layer.
         * 
         * @param feeding
         * @param system system of layers or null
         */
        public void addFeedingLayer(NodeLayer feeding, List<NodeLayer> system)
        {
            feedingLayers.add(feeding);
            feeding.receivingLayers.add(this);
            int newFeedingLayerIndex = transmissionWeights.length;
            transmissionWeights = Arrays.copyOf(transmissionWeights, newFeedingLayerIndex + 1);
            transmissionWeights[newFeedingLayerIndex] = new double[nodeValues.length][feeding.nodeValues.length];
            resetFeedingWeights(newFeedingLayerIndex, true);
            if (system != null)
            {
                int feedingInd = system.indexOf(feeding);
                if (feedingInd < 0)
                {
                    system.add(0, feeding);
                    feedingInd = 0;
                }
                int ind = system.indexOf(this);
                if (ind < 0)
                {
                    system.add(feedingInd + 1, this);
                }
                
            }
        }
        
        public void resetFeedingWeights(int feedingLayerIndex, boolean resetAllFeedingWeights)
        {
            double glorot = glorotBengioWeightFactor(transferFunction);
            
            for(int layer = 0; layer < feedingLayers.size(); layer++)
                if (resetAllFeedingWeights || feedingLayerIndex == layer)
                    for(int i = 0; i < nodeValues.length; i++)
                        Prob.fillFromU(transmissionWeights[layer][i], -0.5 * glorot, 0.5 * glorot);
            
        }
        
        // Glorot & Bengio [2010]
        public double glorotBengioWeightFactor(Dd transferFunction)
        {
            if (0 == feedingLayers.size())
                return 1.0;
            int x = nodeValues.length;
            for(NodeLayer feeding : feedingLayers) { x += feeding.nodeValues.length; }
            double factor = 
                    2.0 *
                    Math.sqrt(6.0) 
                    / Math.sqrt(x)
                    ;
            if (Fn.Composing.underlyingFunction(transferFunction, 0) == Fn.Transfer.sigmoid)
                factor *= 4.0;
            return factor;
        }
        
        
        
        
        
        public void feedForward()
        {
            if (0 == feedingLayers.size())
            {
                return;
            }
            for (int i = 0; i < nodeValues.length; i++)
            {
                double nodeValue = nodeValues[i];
                double input = inputFunction.f(this, i);
                double newValue =  (gateFunction != null ?
                        gateFunction.f(input, nodeValue):
                        input);
                newValue = transferFunction.f(newValue);
//                if (modulationFunction != null)
//                {
//                    value = modulationFunction.f(value, nodeValue);
//                }
                nodeValues[i] = newValue;
            }
        }
        
        public void backpropagate(double[] errorTerms, double[] learningRateCoefs)
        {
            // deltaWij = coef*delta_j*o_i
            //     (for output layer)
            //          = coef * (f'(a_j)(t_j - o_j)) * o_i
            //     (for hidden layers)
            //          = coef * (f'(a_j) * sum(delta_k * w_jk)) * o_i
            
            double[] corrections = new double[nodeValues.length];
            for (int i = 0; i < corrections.length; i++)
            {
                // compute and store:  coef * (k*o_j*(1-o_j)) * (t_j - o_j)
                errorDeltas[i] = (deltaFunction.f(nodeValues[i]) * (errorTerms[i]));
                corrections[i] = learningRateCoefs[i] * errorDeltas[i];
            }
            for(int k = 0; k < feedingLayers.size(); k++)
            {
                NodeLayer feeding = feedingLayers.get(k);
                for (int j = 0; j < feeding.nodeValues.length; j++)
                    for(int i = 0; i < nodeValues.length; i++)
                    {
                        double newWeight = transmissionWeights[k][i][j] + corrections[i] * feeding.nodeValues[j];
                        if (Double.isNaN(newWeight) || Double.isInfinite(newWeight))
                        {
                            newWeight = Math.random() - 0.5;
                        }
                        transmissionWeights[k][i][j] = newWeight;
                    }
            }
        }
        
        public double[] computeHiddenLayerErrorTerms()
        {
            double[] hiddenErrors = Doubles.fill(nodeValues.length, 0);
            for(int k = 0; k < receivingLayers.size(); k++)
            {
                NodeLayer receiving = receivingLayers.get(k);
                int feedingIndex = receiving.feedingLayers.indexOf(this);
                for(int j = 0; j < hiddenErrors.length; j++)
                    for(int i = 0; i < receiving.errorTerms.length; i++)
                        hiddenErrors[j] += receiving.transmissionWeights[feedingIndex][i][j] * receiving.errorTerms[i];
            }
            errorTerms = hiddenErrors;
            return hiddenErrors;
        }
        
        public double[] getModulatedOutput()
        {
            double[] res = Arrays.copyOf(nodeValues, nodeValues.length);
            if (modulationFunction == null)
                return res;
            for(int i = 0; i < res.length; i++)
                res[i] = modulationFunction.f(i, res[i]);
            return res;
        }

        public void print(PrintStream out)
        {
            out.println("N: " + Arrays.toString(nodeValues));
            for(int k = 0; k < transmissionWeights.length; k++)
            {
                out.println("->");
                for (int i = 0; i < nodeValues.length; i++) {
                    out.println(" " + i + ":" + Arrays.toString(transmissionWeights[k][i]));
                }
            }
        }
    }
}

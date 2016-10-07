package oh.neural;

/**
 *
 * @author Paavo Toivanen https://github.com/pvto
 */
public class DDArray {

    double[][] sources;
    public int offsetY = 0;
    public int offset = 0;
    public int size = 0;
    
    public DDArray(double[][] sources)
    {
        this.sources = sources;
        for(int i = 0; i < sources.length; i++)
            size += sources[i].length;
    }
    
    public double next()
    {
        double d = sources[offsetY][offset++];
        if (sources[offsetY].length == offset)
        {
            offsetY++;
            offset = 0;
        }
        return d;
    }
    
    public void reset()
    {
        offsetY = 0;
        offset = 0;
    }
    
}

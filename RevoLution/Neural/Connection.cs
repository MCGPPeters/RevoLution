namespace RevoLution.Neural;

public class Connection
{
    public int FromNodeId { get; }
    public int ToNodeId { get; }
    public double Weight { get; set; }
    public bool Enabled { get; set; }

    // For evolutionary tracking
    public int InnovationNumber { get; }

    public Connection(int fromNodeId, int toNodeId, double weight, int innovationNumber, bool enabled = true)
    {
        FromNodeId = fromNodeId;
        ToNodeId = toNodeId;
        Weight = weight;
        InnovationNumber = innovationNumber;
        Enabled = enabled;
    }

    public Connection Clone()
    {
        return new Connection(FromNodeId, ToNodeId, Weight, InnovationNumber, Enabled);
    }
}

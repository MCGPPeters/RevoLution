using System;

namespace RevoLution.Neural;

public enum NodeType
{
    Input,
    Hidden,
    Output
}

public class Node
{
    public int Id { get; }
    public NodeType Type { get; }
    public double Value { get; set; }
    public double Bias { get; set; }
    public string ActivationFunction { get; set; }

    // For evolutionary tracking
    public int InnovationNumber { get; }
    
    // For backpropagation
    public double Gradient { get; set; }

    public Node(int id, NodeType type, int innovationNumber, string activationFunction = "Sigmoid")
    {
        Id = id;
        Type = type;
        InnovationNumber = innovationNumber;
        Value = 0;
        Bias = 0;
        ActivationFunction = activationFunction;
    }

    public double Activate(double x)
    {
        return ActivationFunction switch
        {
            "Sigmoid" => 1.0 / (1.0 + Math.Exp(-x)),
            "Tanh" => Math.Tanh(x),
            "ReLU" => Math.Max(0, x),
            "LeakyReLU" => x > 0 ? x : 0.01 * x,
            _ => 1.0 / (1.0 + Math.Exp(-x)) // Default to sigmoid
        };
    }    public double ActivateDerivative(double x)
    {
        switch (ActivationFunction)
        {
            case "Sigmoid":
                var sigmoid = 1.0 / (1.0 + Math.Exp(-x));
                return sigmoid * (1 - sigmoid);
            case "Tanh":
                return 1 - Math.Pow(Math.Tanh(x), 2);
            case "ReLU":
                return x > 0 ? 1 : 0;
            case "LeakyReLU":
                return x > 0 ? 1 : 0.01;
            default:
                var defaultSigmoid = 1.0 / (1.0 + Math.Exp(-x));
                return defaultSigmoid * (1 - defaultSigmoid);
        }
    }
}

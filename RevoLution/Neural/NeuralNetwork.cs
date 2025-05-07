using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RevoLution.Neural;

public class NeuralNetwork
{
    private readonly Dictionary<int, Node> _nodes = new();
    private readonly Dictionary<int, List<Connection>> _connections = new();
    private readonly List<int> _inputNodeIds = new();
    private readonly List<int> _outputNodeIds = new();
    private readonly Random _random = new();

    public int SpeciesId { get; set; }
    public double Fitness { get; set; }
    public double AdjustedFitness { get; set; }
    public bool HasBeenEvaluated { get; set; }

    // For tracking changes to topology
    public int Generation { get; set; }
    
    // For caching computed gradients during backprop
    private readonly ConcurrentDictionary<(int, int), double> _gradientCache = new();

    public NeuralNetwork() { }

    public NeuralNetwork(IEnumerable<Node> nodes, IEnumerable<Connection> connections)
    {
        foreach (var node in nodes)
        {
            _nodes[node.Id] = node;
            if (node.Type == NodeType.Input)
                _inputNodeIds.Add(node.Id);
            else if (node.Type == NodeType.Output)
                _outputNodeIds.Add(node.Id);
        }

        foreach (var connection in connections)
        {
            if (!_connections.ContainsKey(connection.FromNodeId))
                _connections[connection.FromNodeId] = new List<Connection>();
            
            _connections[connection.FromNodeId].Add(connection);
        }
    }

    public void AddNode(Node node)
    {
        _nodes[node.Id] = node;
        
        if (node.Type == NodeType.Input)
            _inputNodeIds.Add(node.Id);
        else if (node.Type == NodeType.Output)
            _outputNodeIds.Add(node.Id);
    }

    public void AddConnection(Connection connection)
    {
        if (!_connections.ContainsKey(connection.FromNodeId))
            _connections[connection.FromNodeId] = new List<Connection>();
        
        _connections[connection.FromNodeId].Add(connection);
    }    public List<double> FeedForward(List<double> inputs)
    {
        // Handle the case where the network has no inputs
        if (_inputNodeIds.Count == 0)
        {
            Console.WriteLine("Warning: Network has no input nodes. Creating default inputs.");
            // Return a default output appropriate for the number of output nodes
            return Enumerable.Repeat(0.5, _outputNodeIds.Count).ToList();
        }

        if (inputs.Count != _inputNodeIds.Count)
            throw new ArgumentException($"Expected {_inputNodeIds.Count} inputs, but got {inputs.Count}");

        // Reset all node values
        foreach (var node in _nodes.Values)
        {
            node.Value = 0;
        }

        // Set input values
        for (int i = 0; i < _inputNodeIds.Count; i++)
        {
            _nodes[_inputNodeIds[i]].Value = inputs[i];
        }

        // Sort nodes by layer to ensure proper feed-forward (simple topological sort)
        var nodeLayerOrder = GetNodeEvaluationOrder();

        // Compute node values layer by layer
        foreach (var nodeId in nodeLayerOrder)
        {
            var node = _nodes[nodeId];
            
            if (node.Type == NodeType.Input)
                continue; // Input values are already set
            
            // Sum weighted inputs to this node
            double sum = node.Bias;
            
            foreach (var fromNodeId in _nodes.Keys)
            {
                if (_connections.ContainsKey(fromNodeId))
                {
                    foreach (var connection in _connections[fromNodeId])
                    {
                        if (connection.ToNodeId == nodeId && connection.Enabled)
                        {
                            sum += _nodes[connection.FromNodeId].Value * connection.Weight;
                        }
                    }
                }
            }
            
            // Apply activation function
            node.Value = node.Activate(sum);
        }

        // Return output values
        return _outputNodeIds.Select(id => _nodes[id].Value).ToList();
    }    private List<int> GetNodeEvaluationOrder()
    {
        var nodeTypes = _nodes.ToDictionary(
            kvp => kvp.Key, 
            kvp => kvp.Value.Type
        );

        var result = new List<int>();
        
        // Check if we have a properly formed network
        if (_inputNodeIds.Count == 0 || _outputNodeIds.Count == 0)
        {
            // Return a default order with whatever nodes we have
            result.AddRange(_nodes.Keys);
            return result;
        }
        
        // Add input nodes first
        result.AddRange(_inputNodeIds);
        
        // Add hidden nodes in a valid order (simple algorithm for feed-forward networks)
        var hiddenNodeIds = _nodes.Values
            .Where(n => n.Type == NodeType.Hidden)
            .Select(n => n.Id)
            .ToList();
        
        // Simple approach: try to find nodes with all dependencies resolved
        while (hiddenNodeIds.Count > 0)
        {
            bool progress = false;
            
            for (int i = hiddenNodeIds.Count - 1; i >= 0; i--)
            {
                int nodeId = hiddenNodeIds[i];
                bool allDependenciesResolved = true;
                
                // Check if all nodes this node depends on are already in the result
                foreach (var fromNodeId in _nodes.Keys)
                {
                    if (_connections.ContainsKey(fromNodeId))
                    {
                        foreach (var connection in _connections[fromNodeId])
                        {
                            if (connection.ToNodeId == nodeId && 
                                connection.Enabled && 
                                !result.Contains(connection.FromNodeId))
                            {
                                allDependenciesResolved = false;
                                break;
                            }
                        }
                    }
                    
                    if (!allDependenciesResolved)
                        break;
                }
                
                if (allDependenciesResolved)
                {
                    result.Add(nodeId);
                    hiddenNodeIds.RemoveAt(i);
                    progress = true;
                }
            }
            
            if (!progress && hiddenNodeIds.Count > 0)
            {
                // If we couldn't resolve any nodes but still have nodes left,
                // there might be a cycle. Just add one node to break the cycle.
                result.Add(hiddenNodeIds[0]);
                hiddenNodeIds.RemoveAt(0);
            }
        }
        
        // Add output nodes last
        result.AddRange(_outputNodeIds);
        
        return result;
    }

    public void Backpropagate(List<double> inputs, List<double> targets, double learningRate = 0.01)
    {
        if (targets.Count != _outputNodeIds.Count)
            throw new ArgumentException($"Expected {_outputNodeIds.Count} targets, but got {targets.Count}");

        // Clear gradient cache
        _gradientCache.Clear();
        
        // Forward pass to compute node activations
        var outputs = FeedForward(inputs);
        
        // Calculate output node gradients
        for (int i = 0; i < _outputNodeIds.Count; i++)
        {
            var outputNode = _nodes[_outputNodeIds[i]];
            var error = outputs[i] - targets[i];
            outputNode.Gradient = error;
        }
        
        // Backpropagate gradients in reverse evaluation order
        var nodeOrder = GetNodeEvaluationOrder();
        nodeOrder.Reverse();
        
        foreach (var nodeId in nodeOrder)
        {
            if (_nodes[nodeId].Type == NodeType.Input)
                continue; // No gradients for input nodes
            
            // Find outgoing connections
            var outgoingConnections = GetOutgoingConnections(nodeId);
            
            // Sum the gradients coming from nodes this node connects to
            double gradientSum = 0;
            foreach (var conn in outgoingConnections)
            {
                gradientSum += _nodes[conn.ToNodeId].Gradient * conn.Weight;
            }
            
            // Set the gradient for this node
            var node = _nodes[nodeId];
            double nodeInput = CalculateNodeInput(nodeId);
            double deltaError = gradientSum * node.ActivateDerivative(nodeInput);
            node.Gradient = deltaError;
            
            // Update weights for incoming connections
            var incomingConnections = GetIncomingConnections(nodeId);
            foreach (var conn in incomingConnections)
            {
                var fromNode = _nodes[conn.FromNodeId];
                
                // Weight update: learning_rate * gradient * input_activation
                double weightGradient = node.Gradient * fromNode.Value;
                conn.Weight -= learningRate * weightGradient;
            }
            
            // Update node bias
            node.Bias -= learningRate * node.Gradient;
        }
    }

    private double CalculateNodeInput(int nodeId)
    {
        double sum = _nodes[nodeId].Bias;
        foreach (var fromNodeId in _nodes.Keys)
        {
            if (_connections.ContainsKey(fromNodeId))
            {
                foreach (var connection in _connections[fromNodeId])
                {
                    if (connection.ToNodeId == nodeId && connection.Enabled)
                    {
                        sum += _nodes[connection.FromNodeId].Value * connection.Weight;
                    }
                }
            }
        }
        return sum;
    }

    private List<Connection> GetIncomingConnections(int nodeId)
    {
        var result = new List<Connection>();
        foreach (var fromNodeId in _connections.Keys)
        {
            foreach (var conn in _connections[fromNodeId])
            {
                if (conn.ToNodeId == nodeId && conn.Enabled)
                {
                    result.Add(conn);
                }
            }
        }
        return result;
    }

    private List<Connection> GetOutgoingConnections(int nodeId)
    {
        if (!_connections.ContainsKey(nodeId))
            return new List<Connection>();
            
        return _connections[nodeId]
            .Where(c => c.Enabled)
            .ToList();
    }

    public double CalculateLoss(List<double> outputs, List<double> targets)
    {
        if (outputs.Count != targets.Count)
            throw new ArgumentException("Outputs and targets must have the same length");
        
        double sum = 0;
        for (int i = 0; i < outputs.Count; i++)
        {
            double error = outputs[i] - targets[i];
            sum += error * error; // Squared error
        }
        
        return sum / outputs.Count; // Mean squared error
    }

    public NeuralNetwork Clone()
    {
        var clonedNodes = _nodes.Values.Select(n => 
            new Node(n.Id, n.Type, n.InnovationNumber, n.ActivationFunction) 
            { 
                Bias = n.Bias 
            }
        );
        
        var clonedConnections = new List<Connection>();
        foreach (var fromId in _connections.Keys)
        {
            foreach (var conn in _connections[fromId])
            {
                clonedConnections.Add(conn.Clone());
            }
        }
        
        var clone = new NeuralNetwork(clonedNodes, clonedConnections)
        {
            SpeciesId = SpeciesId,
            Fitness = Fitness,
            AdjustedFitness = AdjustedFitness,
            Generation = Generation
        };
        
        return clone;
    }

    // Utility methods to get network information
    public int GetInputCount() => _inputNodeIds.Count;
    public int GetOutputCount() => _outputNodeIds.Count;
    public int GetHiddenNodeCount() => _nodes.Count - _inputNodeIds.Count - _outputNodeIds.Count;
}

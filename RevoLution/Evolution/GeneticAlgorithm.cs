using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RevoLution.Evolution;

using RevoLution.Neural;

public class GeneticAlgorithm
{
    private readonly Random _random = new();
    private readonly List<Species> _species = new();
    private readonly Dictionary<(int FromNode, int ToNode), int> _innovationHistory = new();
    private int _nextInnovationNumber = 0;
    private int _nextSpeciesId = 0;
    private int _nextNodeId = 0;

    // Configuration
    public int PopulationSize { get; set; } = 100;
    public double MutateWeightChance { get; set; } = 0.8;
    public double AddNodeChance { get; set; } = 0.03;
    public double AddConnectionChance { get; set; } = 0.05;
    public double WeightPerturbChance { get; set; } = 0.9;
    public double WeightPerturbAmount { get; set; } = 0.1;
    public double CrossoverChance { get; set; } = 0.75;
    public double DisableChance { get; set; } = 0.75;
    public double CompatibilityThreshold { get; set; } = 3.0;
    public double CompatibilityWeightCoefficient { get; set; } = 0.4;
    public double CompatibilityDisjointCoefficient { get; set; } = 1.0;
    public double CompatibilityExcessCoefficient { get; set; } = 1.0;
    public int StagnationThreshold { get; set; } = 15;
    public double SurvivalThreshold { get; set; } = 0.2;
      // Generation tracking
    public int Generation { get; private set; } = 0;
    public NeuralNetwork? BestNetwork { get; private set; }
    public double BestFitness { get; private set; }

    public GeneticAlgorithm() { }

    public List<NeuralNetwork> InitializePopulation(int inputSize, int outputSize, int hiddenNodes = 0)
    {
        var population = new List<NeuralNetwork>();

        for (int i = 0; i < PopulationSize; i++)
        {
            var network = CreateNetwork(inputSize, outputSize, hiddenNodes);
            population.Add(network);
        }

        return population;
    }

    private NeuralNetwork CreateNetwork(int inputSize, int outputSize, int hiddenNodes)
    {
        var nodes = new List<Node>();
        var connections = new List<Connection>();

        // Create input nodes
        for (int i = 0; i < inputSize; i++)
        {
            nodes.Add(new Node(_nextNodeId++, NodeType.Input, GetNextInnovationNumber()));
        }

        // Create hidden nodes
        for (int i = 0; i < hiddenNodes; i++)
        {
            nodes.Add(new Node(_nextNodeId++, NodeType.Hidden, GetNextInnovationNumber()));
        }

        // Create output nodes
        for (int i = 0; i < outputSize; i++)
        {
            nodes.Add(new Node(_nextNodeId++, NodeType.Output, GetNextInnovationNumber()));
        }

        // Create connections
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = inputSize + hiddenNodes; j < inputSize + hiddenNodes + outputSize; j++)
            {
                connections.Add(new Connection(
                    i,
                    j,
                    RandomWeight(),
                    GetInnovationNumber(i, j)
                ));
            }
        }

        // If we have hidden nodes, create connections to them
        if (hiddenNodes > 0)
        {
            for (int i = 0; i < inputSize; i++)
            {
                for (int h = inputSize; h < inputSize + hiddenNodes; h++)
                {
                    connections.Add(new Connection(
                        i,
                        h,
                        RandomWeight(),
                        GetInnovationNumber(i, h)
                    ));
                }
            }

            for (int h = inputSize; h < inputSize + hiddenNodes; h++)
            {
                for (int o = inputSize + hiddenNodes; o < inputSize + hiddenNodes + outputSize; o++)
                {
                    connections.Add(new Connection(
                        h,
                        o,
                        RandomWeight(),
                        GetInnovationNumber(h, o)
                    ));
                }
            }
        }

        return new NeuralNetwork(nodes, connections);
    }

    public void EvaluatePopulation(List<NeuralNetwork> population, Func<NeuralNetwork, double> fitnessFunction)
    {
        Parallel.ForEach(population, network =>
        {
            network.Fitness = fitnessFunction(network);
            network.HasBeenEvaluated = true;
        });

        UpdateBestNetwork(population);
    }

    private void UpdateBestNetwork(List<NeuralNetwork> population)
    {
        var bestInGeneration = population.OrderByDescending(n => n.Fitness).First();
        
        if (BestNetwork == null || bestInGeneration.Fitness > BestFitness)
        {
            BestNetwork = bestInGeneration.Clone();
            BestFitness = bestInGeneration.Fitness;
        }
    }

    public List<NeuralNetwork> Evolve(List<NeuralNetwork> currentPopulation)
    {
        Generation++;
        
        // Ensure all networks have been evaluated
        if (currentPopulation.Any(n => !n.HasBeenEvaluated))
        {
            throw new InvalidOperationException("All networks must be evaluated before evolution");
        }

        // Speciate
        SpeciatePopulation(currentPopulation);
        
        // Calculate adjusted fitness
        foreach (var species in _species)
        {
            species.CalculateAdjustedFitness();
        }
        
        // Check for stagnation and remove stagnant species
        HandleStagnation();
        
        // Breed new population
        var newPopulation = BreedNewPopulation();
        
        // Update species representatives for next generation
        foreach (var species in _species)
        {
            species.UpdateRepresentative();
        }
        
        return newPopulation;
    }

    private void SpeciatePopulation(List<NeuralNetwork> population)
    {
        // Clear existing species members but keep representatives
        foreach (var species in _species)
        {
            species.Networks.Clear();
        }
        
        // Add each network to a species
        foreach (var network in population)
        {
            bool foundSpecies = false;
            
            foreach (var species in _species)
            {
                if (CalculateCompatibilityDistance(network, species.Representative) <= CompatibilityThreshold)
                {
                    species.Networks.Add(network);
                    network.SpeciesId = species.Id;
                    foundSpecies = true;
                    break;
                }
            }
            
            if (!foundSpecies)
            {
                // Create new species with this network as representative
                var newSpecies = new Species(_nextSpeciesId++, network);
                _species.Add(newSpecies);
            }
        }
        
        // Remove empty species
        _species.RemoveAll(s => s.Networks.Count == 0);
    }

    private double CalculateCompatibilityDistance(NeuralNetwork network1, NeuralNetwork network2)
    {
        // Get all connections from both networks
        var connections1 = GetAllConnections(network1);
        var connections2 = GetAllConnections(network2);
        
        // Sort by innovation number
        connections1.Sort((a, b) => a.InnovationNumber.CompareTo(b.InnovationNumber));
        connections2.Sort((a, b) => a.InnovationNumber.CompareTo(b.InnovationNumber));
        
        int disjointCount = 0;
        int excessCount = 0;
        double weightDifference = 0;
        int matchingCount = 0;
        
        int i = 0, j = 0;
        
        while (i < connections1.Count && j < connections2.Count)
        {
            Connection conn1 = connections1[i];
            Connection conn2 = connections2[j];
            
            if (conn1.InnovationNumber == conn2.InnovationNumber)
            {
                // Matching genes
                matchingCount++;
                weightDifference += Math.Abs(conn1.Weight - conn2.Weight);
                i++;
                j++;
            }
            else if (conn1.InnovationNumber < conn2.InnovationNumber)
            {
                // Disjoint gene in network1
                disjointCount++;
                i++;
            }
            else
            {
                // Disjoint gene in network2
                disjointCount++;
                j++;
            }
        }
        
        // Excess genes
        excessCount = (connections1.Count - i) + (connections2.Count - j);
        
        // Calculate normalized distance
        int n = Math.Max(connections1.Count, connections2.Count);
        if (n == 0) n = 1; // Avoid division by zero
        
        double averageWeightDiff = matchingCount > 0 ? weightDifference / matchingCount : 0;
        
        return (CompatibilityExcessCoefficient * excessCount / n) +
               (CompatibilityDisjointCoefficient * disjointCount / n) +
               (CompatibilityWeightCoefficient * averageWeightDiff);
    }

    private List<Connection> GetAllConnections(NeuralNetwork network)
    {
        var result = new List<Connection>();
        
        // We need to reflect or modify the NeuralNetwork class to expose this data
        // For now, we'll assume we can access it through a method we'll add later
        
        // This is a placeholder - implementation will depend on the NeuralNetwork class
        var nodesAndConnections = GetNodesAndConnections(network);
        var connections = nodesAndConnections.Item2;
        
        return connections;
    }

    private (List<Node>, List<Connection>) GetNodesAndConnections(NeuralNetwork network)
    {
        // This is a placeholder - we would need to implement proper access in the NeuralNetwork class
        // For now, let's assume we can access them through reflection or by modifying the class
        
        // In real implementation, would access network._nodes and network._connections
        return (new List<Node>(), new List<Connection>());
    }

    private void HandleStagnation()
    {
        foreach (var species in _species)
        {
            double currentBestFitness = species.Networks.Max(n => n.Fitness);
            
            if (currentBestFitness > species.BestFitness)
            {
                species.BestFitness = currentBestFitness;
                species.StagnationCount = 0;
            }
            else
            {
                species.StagnationCount++;
            }
        }
        
        // Remove stagnant species, but keep at least one
        if (_species.Count > 1)
        {
            _species.RemoveAll(s => s.StagnationCount >= StagnationThreshold);
            
            // Ensure we still have at least one species
            if (_species.Count == 0)
            {
                // If all species were stagnant, keep the one with highest fitness
                var bestSpecies = _species.OrderByDescending(s => s.BestFitness).First();
                _species.Clear();
                _species.Add(bestSpecies);
            }
        }
    }

    private List<NeuralNetwork> BreedNewPopulation()
    {
        var newPopulation = new List<NeuralNetwork>();
        
        // Calculate total adjusted fitness
        double totalAdjustedFitness = _species.Sum(s => s.Networks.Sum(n => n.AdjustedFitness));
        
        // Calculate how many offspring each species should have
        foreach (var species in _species)
        {
            double speciesAdjustedFitness = species.Networks.Sum(n => n.AdjustedFitness);
            int offspringCount = (int)Math.Floor((speciesAdjustedFitness / totalAdjustedFitness) * PopulationSize);
            
            if (offspringCount > 0)
            {
                // Keep top performing networks (elitism)
                int eliteCount = (int)Math.Max(1, Math.Floor(offspringCount * SurvivalThreshold));
                var elites = species.Networks
                    .OrderByDescending(n => n.Fitness)
                    .Take(eliteCount)
                    .Select(n => n.Clone())
                    .ToList();
                
                newPopulation.AddRange(elites);
                
                // Breed remaining offspring
                for (int i = eliteCount; i < offspringCount; i++)
                {
                    var child = BreedChild(species);
                    newPopulation.Add(child);
                }
            }
        }
        
        // Fill remaining population slots with random offspring
        while (newPopulation.Count < PopulationSize)
        {
            // Select a species with probability proportional to its adjusted fitness
            var selectedSpecies = SelectSpeciesByFitness(totalAdjustedFitness);
            var child = BreedChild(selectedSpecies);
            newPopulation.Add(child);
        }
        
        return newPopulation;
    }

    private Species SelectSpeciesByFitness(double totalAdjustedFitness)
    {
        double targetValue = _random.NextDouble() * totalAdjustedFitness;
        double currentSum = 0;
        
        foreach (var species in _species)
        {
            currentSum += species.Networks.Sum(n => n.AdjustedFitness);
            if (currentSum >= targetValue)
            {
                return species;
            }
        }
        
        // Fallback to the last species
        return _species.Last();
    }

    private NeuralNetwork BreedChild(Species species)
    {
        NeuralNetwork child;
        
        if (_random.NextDouble() < CrossoverChance && species.Networks.Count > 1)
        {
            // Select two parents from the species
            var parent1 = SelectNetworkByFitness(species.Networks);
            var parent2 = SelectNetworkByFitness(species.Networks);
            
            // Ensure parents are different
            while (parent2 == parent1)
            {
                parent2 = SelectNetworkByFitness(species.Networks);
            }
            
            // Crossover
            child = Crossover(parent1, parent2);
        }
        else
        {
            // Asexual reproduction - clone a single parent
            var parent = SelectNetworkByFitness(species.Networks);
            child = parent.Clone();
        }
        
        // Mutate the child
        Mutate(child);
        
        child.SpeciesId = species.Id;
        child.HasBeenEvaluated = false;
        child.Generation = Generation;
        
        return child;
    }

    private NeuralNetwork SelectNetworkByFitness(List<NeuralNetwork> networks)
    {
        // Tournament selection
        const int tournamentSize = 3;
        var competitors = new List<NeuralNetwork>();
        
        for (int i = 0; i < tournamentSize; i++)
        {
            competitors.Add(networks[_random.Next(networks.Count)]);
        }
        
        return competitors.OrderByDescending(n => n.Fitness).First();
    }

    private NeuralNetwork Crossover(NeuralNetwork parent1, NeuralNetwork parent2)
    {
        // Ensure parent1 has higher fitness
        if (parent2.Fitness > parent1.Fitness)
        {
            (parent1, parent2) = (parent2, parent1);
        }
        
        var nodesP1 = GetNodesAndConnections(parent1).Item1;
        var connectionsP1 = GetNodesAndConnections(parent1).Item2;
        var nodesP2 = GetNodesAndConnections(parent2).Item1;
        var connectionsP2 = GetNodesAndConnections(parent2).Item2;
        
        var childNodes = new List<Node>();
        var childConnections = new List<Connection>();
        
        // Create a hashset of node IDs for quick lookup
        var nodeIds1 = nodesP1.Select(n => n.Id).ToHashSet();
        var nodeIds2 = nodesP2.Select(n => n.Id).ToHashSet();
        
        // Add all nodes from parent1 (the fitter parent)
        foreach (var node in nodesP1)
        {
            childNodes.Add(new Node(
                node.Id, 
                node.Type, 
                node.InnovationNumber, 
                node.ActivationFunction
            ) { 
                Bias = node.Bias 
            });
        }
        
        // Add unique nodes from parent2
        foreach (var node in nodesP2)
        {
            if (!nodeIds1.Contains(node.Id))
            {
                childNodes.Add(new Node(
                    node.Id, 
                    node.Type, 
                    node.InnovationNumber, 
                    node.ActivationFunction
                ) { 
                    Bias = node.Bias 
                });
            }
        }
        
        // Create dictionary of innovation numbers for quick lookup
        var connInnovations1 = connectionsP1.ToDictionary(c => c.InnovationNumber);
        var connInnovations2 = connectionsP2.ToDictionary(c => c.InnovationNumber);
        
        // Get all innovation numbers from both parents
        var allInnovations = new HashSet<int>(
            connectionsP1.Select(c => c.InnovationNumber)
            .Concat(connectionsP2.Select(c => c.InnovationNumber))
        );
        
        // For each innovation number, choose the connection from appropriate parent
        foreach (var innovation in allInnovations)
        {
            if (connInnovations1.TryGetValue(innovation, out var conn1))
            {
                if (connInnovations2.TryGetValue(innovation, out var conn2))
                {
                    // Both parents have this connection - randomly choose one
                    var selectedConn = _random.NextDouble() < 0.5 ? conn1 : conn2;
                    
                    // Check for disabled connections
                    bool isEnabled = selectedConn.Enabled;
                    if (!conn1.Enabled || !conn2.Enabled)
                    {
                        // If either parent has it disabled, chance to disable in child
                        if (_random.NextDouble() < DisableChance)
                        {
                            isEnabled = false;
                        }
                    }
                    
                    childConnections.Add(new Connection(
                        selectedConn.FromNodeId,
                        selectedConn.ToNodeId,
                        selectedConn.Weight,
                        selectedConn.InnovationNumber,
                        isEnabled
                    ));
                }
                else
                {
                    // Only parent1 has this connection - inherit from parent1
                    childConnections.Add(new Connection(
                        conn1.FromNodeId,
                        conn1.ToNodeId,
                        conn1.Weight,
                        conn1.InnovationNumber,
                        conn1.Enabled
                    ));
                }
            }
            else if (connInnovations2.TryGetValue(innovation, out var conn2))
            {
                // Only parent2 has this connection - inherit from parent2
                childConnections.Add(new Connection(
                    conn2.FromNodeId,
                    conn2.ToNodeId,
                    conn2.Weight,
                    conn2.InnovationNumber,
                    conn2.Enabled
                ));
            }
        }
        
        return new NeuralNetwork(childNodes, childConnections);
    }    private void Mutate(NeuralNetwork network)
    {
        // Verify network has proper input and output nodes
        int expectedInputs = 4; // For CartPole environment
        int expectedOutputs = 1;
        
        if (network.GetInputCount() != expectedInputs || network.GetOutputCount() != expectedOutputs)
        {
            Console.WriteLine($"Fixing network with incorrect node counts: Inputs={network.GetInputCount()}, Outputs={network.GetOutputCount()}");
            
            var nodes = GetNodesAndConnections(network).Item1;
            var connections = GetNodesAndConnections(network).Item2;
            var existingNodes = new HashSet<int>(nodes.Select(n => n.Id));
            
            // Keep track of new nodes we add
            var newNodes = new List<Node>();
            var newConnections = new List<Connection>();
            
            // Fix input nodes if needed
            if (network.GetInputCount() != expectedInputs)
            {
                // Create the correct number of input nodes
                for (int i = 0; i < expectedInputs; i++)
                {
                    var inputNode = new Node(_nextNodeId++, NodeType.Input, GetNextInnovationNumber());
                    newNodes.Add(inputNode);
                }
            }
            
            // Fix output nodes if needed
            if (network.GetOutputCount() != expectedOutputs)
            {
                // Create the correct number of output nodes
                for (int i = 0; i < expectedOutputs; i++)
                {
                    var outputNode = new Node(_nextNodeId++, NodeType.Output, GetNextInnovationNumber());
                    newNodes.Add(outputNode);
                }
            }
            
            // Add connections between new input and output nodes
            var inputNodes = newNodes.Where(n => n.Type == NodeType.Input).ToList();
            var outputNodes = newNodes.Where(n => n.Type == NodeType.Output).ToList();
            
            foreach (var input in inputNodes)
            {
                foreach (var output in outputNodes)
                {
                    var connection = new Connection(
                        input.Id,
                        output.Id,
                        RandomWeight(),
                        GetInnovationNumber(input.Id, output.Id)
                    );
                    newConnections.Add(connection);
                }
            }
            
            // Create a new network with the fixed structure
            var fixedNetwork = new NeuralNetwork(newNodes, newConnections);
            
            // Copy the network's properties
            fixedNetwork.Fitness = network.Fitness;
            fixedNetwork.AdjustedFitness = network.AdjustedFitness;
            fixedNetwork.SpeciesId = network.SpeciesId;
            fixedNetwork.Generation = network.Generation;
            
            // Replace the old network with the fixed one
            // We need to copy all properties to the original network
            foreach (var node in newNodes)
            {
                network.AddNode(node);
            }
            
            foreach (var connection in newConnections)
            {
                network.AddConnection(connection);
            }
            
            return; // Skip further mutations for this network
        }
        
        // Mutate connection weights
        if (_random.NextDouble() < MutateWeightChance)
        {
            MutateConnectionWeights(network);
        }
        
        // Add a new node
        if (_random.NextDouble() < AddNodeChance)
        {
            AddNode(network);
        }
        
        // Add a new connection
        if (_random.NextDouble() < AddConnectionChance)
        {
            AddConnection(network);
        }
    }

    private void MutateConnectionWeights(NeuralNetwork network)
    {
        var connections = GetNodesAndConnections(network).Item2;
        
        foreach (var connection in connections)
        {
            if (_random.NextDouble() < WeightPerturbChance)
            {
                // Perturb the weight slightly
                connection.Weight += (_random.NextDouble() * 2 - 1) * WeightPerturbAmount;
            }
            else
            {
                // Assign a completely new weight
                connection.Weight = RandomWeight();
            }
        }
    }

    private void AddNode(NeuralNetwork network)
    {
        var connections = GetNodesAndConnections(network).Item2;
        
        if (connections.Count == 0)
            return;
        
        // Select a random enabled connection to split
        var enabledConnections = connections.Where(c => c.Enabled).ToList();
        if (enabledConnections.Count == 0)
            return;
            
        var connectionToSplit = enabledConnections[_random.Next(enabledConnections.Count)];
        
        // Disable the selected connection
        connectionToSplit.Enabled = false;
        
        // Create a new node
        int newNodeId = _nextNodeId++;
        var newNode = new Node(newNodeId, NodeType.Hidden, GetNextInnovationNumber());
        
        // Add the new node
        network.AddNode(newNode);
        
        // Create two new connections
        var inConnection = new Connection(
            connectionToSplit.FromNodeId,
            newNodeId,
            1.0, // Weight of 1 to the new node
            GetInnovationNumber(connectionToSplit.FromNodeId, newNodeId)
        );
        
        var outConnection = new Connection(
            newNodeId,
            connectionToSplit.ToNodeId,
            connectionToSplit.Weight, // Preserve the old weight
            GetInnovationNumber(newNodeId, connectionToSplit.ToNodeId)
        );
        
        // Add the new connections
        network.AddConnection(inConnection);
        network.AddConnection(outConnection);
    }

    private void AddConnection(NeuralNetwork network)
    {
        var nodes = GetNodesAndConnections(network).Item1;
        var connections = GetNodesAndConnections(network).Item2;
        
        // Get all possible connections
        var existingConnections = new HashSet<(int from, int to)>();
        foreach (var conn in connections)
        {
            existingConnections.Add((conn.FromNodeId, conn.ToNodeId));
        }
        
        // Find all possible new connections
        var possibleConnections = new List<(int from, int to)>();
        
        foreach (var fromNode in nodes)
        {
            if (fromNode.Type == NodeType.Output)
                continue; // Outputs can't be source nodes in a feedforward network
                
            foreach (var toNode in nodes)
            {
                if (toNode.Type == NodeType.Input)
                    continue; // Inputs can't be destination nodes
                    
                if (fromNode.Id == toNode.Id)
                    continue; // No self-connections
                    
                // Avoid creating cycles (for simplicity, assume no connections from higher index to lower index for hidden nodes)
                if (fromNode.Type == NodeType.Hidden && toNode.Type == NodeType.Hidden)
                {
                    if (fromNode.Id >= toNode.Id)
                        continue;
                }
                
                // Check if connection already exists
                if (!existingConnections.Contains((fromNode.Id, toNode.Id)))
                {
                    possibleConnections.Add((fromNode.Id, toNode.Id));
                }
            }
        }
        
        // If no possible new connections, return
        if (possibleConnections.Count == 0)
            return;
            
        // Choose a random new connection to add
        var newConn = possibleConnections[_random.Next(possibleConnections.Count)];
        
        // Create and add the new connection
        var connection = new Connection(
            newConn.from,
            newConn.to,
            RandomWeight(),
            GetInnovationNumber(newConn.from, newConn.to)
        );
        
        network.AddConnection(connection);
    }

    private double RandomWeight()
    {
        return (_random.NextDouble() * 4) - 2; // Random weight between -2 and 2
    }

    private int GetNextInnovationNumber()
    {
        return _nextInnovationNumber++;
    }

    private int GetInnovationNumber(int fromNode, int toNode)
    {
        var key = (fromNode, toNode);
        
        if (!_innovationHistory.ContainsKey(key))
        {
            _innovationHistory[key] = _nextInnovationNumber++;
        }
        
        return _innovationHistory[key];
    }
}

using System;
using System.Collections.Generic;
using System.Linq;

namespace RevoLution.Hybrid;

using RevoLution.Neural;
using RevoLution.RL;
using RevoLution.Evolution;
using RevoLution.Novelty;

/// <summary>
/// HybridLearner combines evolutionary approaches (NEAT-style) with reinforcement learning and novelty search.
/// It uses evolution to discover network architectures, novelty search to promote diversity,
/// and reinforcement learning to fine-tune networks for specific tasks.
/// </summary>
public class HybridLearner
{
    private readonly GeneticAlgorithm _geneticAlgorithm;
    private readonly ReinforcementLearner _reinforcementLearner;
    private readonly NoveltySearch _noveltySearch;
    private readonly Random _random = new();
    
    // Configuration
    public int PopulationSize { get => _geneticAlgorithm.PopulationSize; set => _geneticAlgorithm.PopulationSize = value; }
    public double NoveltyWeight { get; set; } = 0.5;
    public double TaskWeight { get; set; } = 0.5;
    public int TrainingEpisodes { get; set; } = 50;
    public int EvaluationEpisodes { get; set; } = 10;
    public int GenerationsPerCycle { get; set; } = 10;
    public double SpecializationThreshold { get; set; } = 0.7;
      // Current state
    public List<NeuralNetwork> Population { get; private set; } = new();
    public Dictionary<int, double[]> BehaviorCharacteristics { get; } = new();
    public Dictionary<int, double> FitnessScores { get; } = new();
    public List<NeuralNetwork> SpecializedNetworks { get; } = new();
    
    public HybridLearner(
        GeneticAlgorithm? geneticAlgorithm = null,
        ReinforcementLearner? reinforcementLearner = null,
        NoveltySearch? noveltySearch = null)
    {
        _geneticAlgorithm = geneticAlgorithm ?? new GeneticAlgorithm();
        _reinforcementLearner = reinforcementLearner ?? new ReinforcementLearner();
        _noveltySearch = noveltySearch ?? new NoveltySearch();
    }    /// <summary>
    /// Initializes the population with random neural networks.
    /// </summary>
    public void Initialize(int inputSize, int outputSize, int initialHiddenNodes = 0)
    {
        Population = _geneticAlgorithm.InitializePopulation(inputSize, outputSize, initialHiddenNodes);
        
        // Verify networks were properly initialized and fix any issues
        for (int i = 0; i < Population.Count; i++)
        {
            var network = Population[i];
            if (network.GetInputCount() != inputSize)
            {
                Console.WriteLine($"Warning: Network initialized with {network.GetInputCount()} inputs instead of {inputSize}. Reinitializing this network.");
                
                // Replace the improperly initialized network
                var newNetwork = CreateBasicNetwork(inputSize, outputSize, initialHiddenNodes);
                Population[i] = newNetwork;
            }
        }
        
        Console.WriteLine($"Initialized population of {Population.Count} networks with {inputSize} inputs, {outputSize} outputs, and {initialHiddenNodes} hidden nodes");
    }
    
    /// <summary>
    /// Creates a basic neural network with the specified dimensions.
    /// </summary>
    private NeuralNetwork CreateBasicNetwork(int inputSize, int outputSize, int hiddenNodes)
    {
        var nodes = new List<Node>();
        var connections = new List<Connection>();
        int nodeId = 0;
        int innovationNumber = 0;
        
        // Create input nodes
        for (int i = 0; i < inputSize; i++)
        {
            nodes.Add(new Node(nodeId++, NodeType.Input, innovationNumber++));
        }
        
        // Create hidden nodes
        for (int i = 0; i < hiddenNodes; i++)
        {
            nodes.Add(new Node(nodeId++, NodeType.Hidden, innovationNumber++));
        }
        
        // Create output nodes
        for (int i = 0; i < outputSize; i++)
        {
            nodes.Add(new Node(nodeId++, NodeType.Output, innovationNumber++));
        }
        
        // Connect inputs to outputs (or to hidden if we have them)
        for (int i = 0; i < inputSize; i++)
        {
            if (hiddenNodes > 0)
            {
                // Connect inputs to hidden nodes
                for (int h = inputSize; h < inputSize + hiddenNodes; h++)
                {
                    connections.Add(new Connection(i, h, RandomWeight(), innovationNumber++));
                }
            }
            else
            {
                // Connect inputs directly to outputs if no hidden nodes
                for (int o = inputSize; o < inputSize + outputSize; o++)
                {
                    connections.Add(new Connection(i, o, RandomWeight(), innovationNumber++));
                }
            }
        }
        
        // Connect hidden to outputs if we have hidden nodes
        if (hiddenNodes > 0)
        {
            for (int h = inputSize; h < inputSize + hiddenNodes; h++)
            {
                for (int o = inputSize + hiddenNodes; o < inputSize + hiddenNodes + outputSize; o++)
                {
                    connections.Add(new Connection(h, o, RandomWeight(), innovationNumber++));
                }
            }
        }
        
        return new NeuralNetwork(nodes, connections);
    }
    
    private double RandomWeight()
    {
        return (_random.NextDouble() * 4) - 2; // Between -2 and 2
    }

    /// <summary>
    /// Runs the hybrid learning algorithm for a specified number of cycles.
    /// Each cycle consists of:
    /// 1. Evolving the population for multiple generations
    /// 2. Identifying specialized networks
    /// 3. Training specialized networks with reinforcement learning
    /// </summary>
    public void Learn(
        Func<NeuralNetwork, (double, double[])> evaluateFunction,
        Func<List<double>, List<double>, (List<double>, double, bool)> environmentFunction,
        int cycles = 10)
    {
        for (int cycle = 0; cycle < cycles; cycle++)
        {
            Console.WriteLine($"Starting cycle {cycle + 1}/{cycles}");
            
            // Evolutionary phase: Run multiple generations of evolution
            for (int gen = 0; gen < GenerationsPerCycle; gen++)
            {
                // Evaluate population with both task performance and novelty
                EvaluatePopulation(evaluateFunction);
                
                // Evolve the population
                Population = _geneticAlgorithm.Evolve(Population);
                
                Console.WriteLine($"  Generation {gen + 1}/{GenerationsPerCycle} - Best Fitness: {FitnessScores.Values.Max():F4}");
            }
            
            // Identify networks that are good candidates for specialization
            var candidates = IdentifySpecializationCandidates();
            
            // Train specialized networks with reinforcement learning
            foreach (var network in candidates)
            {
                Console.WriteLine($"  Training specialized network with fitness {network.Fitness:F4}");
                
                // Clone the network to preserve the original version in the population
                var specializedNetwork = network.Clone();
                
                // Fine-tune with reinforcement learning
                _reinforcementLearner.Train(
                    specializedNetwork,
                    environmentFunction,
                    TrainingEpisodes
                );
                
                // Evaluate the specialized network
                double avgReward = EvaluateRLNetwork(specializedNetwork, environmentFunction);
                specializedNetwork.Fitness = avgReward;
                
                Console.WriteLine($"  Specialized network fitness: {avgReward:F4}");
                
                // Add to collection of specialized networks
                SpecializedNetworks.Add(specializedNetwork);
            }
            
            // Optionally, we could reintroduce specialized networks back into the population
            ReintroduceSpecializedNetworks(candidates.Count);
        }
    }    /// <summary>
    /// Evaluates the fitness of each network in the population.
    /// The fitness combines both task performance and novelty.
    /// </summary>
    private void EvaluatePopulation(Func<NeuralNetwork, (double, double[])> evaluateFunction)
    {
        // Clear previous behavior characteristics
        BehaviorCharacteristics.Clear();
        
        // Collect behaviors and raw fitness for all networks
        for (int i = 0; i < Population.Count; i++)
        {
            var network = Population[i];
            try
            {
                var (rawFitness, behavior) = evaluateFunction(network);
                
                // Store behavior characteristics
                BehaviorCharacteristics[network.GetHashCode()] = behavior;
                
                // Mark as evaluated
                network.HasBeenEvaluated = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error evaluating network: {ex.Message}");
                
                // Try to diagnose the issue
                Console.WriteLine($"Network details: Inputs={network.GetInputCount()}, Outputs={network.GetOutputCount()}, Hidden={network.GetHiddenNodeCount()}");
                
                // Create a replacement network
                Console.WriteLine("Creating replacement network...");
                var newNetwork = CreateBasicNetwork(4, 1, 2); // Hard-coding parameters for quick fix
                Population[i] = newNetwork;
                
                // Try again with the new network
                try
                {
                    var (rawFitness, behavior) = evaluateFunction(newNetwork);
                    BehaviorCharacteristics[newNetwork.GetHashCode()] = behavior;
                    newNetwork.HasBeenEvaluated = true;
                }
                catch (Exception innerEx)
                {
                    Console.WriteLine($"Still failed with replacement network: {innerEx.Message}");
                    // Give this network a very low fitness to ensure it's not selected
                    newNetwork.Fitness = -1000;
                    newNetwork.HasBeenEvaluated = true;
                    BehaviorCharacteristics[newNetwork.GetHashCode()] = new double[4] { 0, 0, 0, 0 };
                }
            }
        }
        
        // Calculate novelty for each network
        var allBehaviors = BehaviorCharacteristics.Values.ToList();
        
        foreach (var network in Population)
        {
            int networkHash = network.GetHashCode();
            
            if (BehaviorCharacteristics.TryGetValue(networkHash, out var behavior))
            {
                try
                {
                    // Get task fitness and novelty score
                    var (rawFitness, _) = evaluateFunction(network);
                    double noveltyScore = _noveltySearch.CalculateNovelty(behavior, allBehaviors);
                    
                    // Combine scores with weighted sum
                    double combinedFitness = (TaskWeight * rawFitness) + (NoveltyWeight * noveltyScore);
                    
                    // Update fitness
                    network.Fitness = combinedFitness;
                    FitnessScores[networkHash] = combinedFitness;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error in final fitness calculation: {ex.Message}");
                    network.Fitness = -1000; // Very low fitness
                    FitnessScores[networkHash] = -1000;
                }
            }
        }
    }

    /// <summary>
    /// Identifies networks that are good candidates for specialization through RL.
    /// </summary>
    private List<NeuralNetwork> IdentifySpecializationCandidates()
    {
        // Sort networks by fitness
        var sortedNetworks = Population
            .OrderByDescending(n => n.Fitness)
            .ToList();
        
        // Take top performers that exceed our specialization threshold
        double maxFitness = sortedNetworks.First().Fitness;
        double threshold = maxFitness * SpecializationThreshold;
        
        var candidates = sortedNetworks
            .Where(n => n.Fitness >= threshold)
            .Take(3) // Limit to top 3 candidates
            .ToList();
        
        return candidates;
    }

    /// <summary>
    /// Evaluates a reinforcement-learned network over multiple episodes.
    /// </summary>
    private double EvaluateRLNetwork(
        NeuralNetwork network,
        Func<List<double>, List<double>, (List<double>, double, bool)> environmentFunction)
    {
        double totalReward = 0;
        
        for (int episode = 0; episode < EvaluationEpisodes; episode++)
        {            // Reset environment
            var initialState = new List<double>();
            var state = environmentFunction(initialState, initialState).Item1;
            double episodeReward = 0;
            bool done = false;
            
            // Run until episode ends
            while (!done)
            {
                // Use network to select action
                var action = network.FeedForward(state);
                
                // Take action in environment
                var (nextState, reward, isDone) = environmentFunction(state, action);
                
                // Update state and accumulate reward
                state = nextState;
                episodeReward += reward;
                done = isDone;
            }
            
            totalReward += episodeReward;
        }
        
        return totalReward / EvaluationEpisodes;
    }

    /// <summary>
    /// Reintroduces specialized networks back into the population,
    /// replacing the lowest-fitness networks.
    /// </summary>
    private void ReintroduceSpecializedNetworks(int replaced)
    {
        // Only reintroduce if we have specialized networks to add
        if (SpecializedNetworks.Count == 0)
            return;
        
        // Sort population by fitness (ascending)
        var sortedPopulation = Population
            .OrderBy(n => n.Fitness)
            .ToList();
        
        // Determine how many networks to replace (at most the number of specialized networks)
        int replaceCount = Math.Min(replaced, SpecializedNetworks.Count);
        
        // Replace lowest-fitness networks with specialized ones
        for (int i = 0; i < replaceCount; i++)
        {
            // Get a random specialized network
            int specializedIndex = _random.Next(SpecializedNetworks.Count);
            var specializedNetwork = SpecializedNetworks[specializedIndex];
            
            // Create a copy of it (to preserve the original)
            var networkCopy = specializedNetwork.Clone();
            
            // Replace a low-fitness network in the population
            int populationIndex = Population.IndexOf(sortedPopulation[i]);
            if (populationIndex >= 0)
            {
                Population[populationIndex] = networkCopy;
            }
        }
    }    /// <summary>
    /// Gets the best specialized network based on fitness.
    /// </summary>
    public NeuralNetwork? GetBestSpecializedNetwork()
    {
        if (SpecializedNetworks.Count == 0)
            return null;
            
        return SpecializedNetworks
            .OrderByDescending(n => n.Fitness)
            .First();
    }

    /// <summary>
    /// Gets the best network in the current population.
    /// </summary>
    public NeuralNetwork GetBestNetwork()
    {
        return Population
            .OrderByDescending(n => n.Fitness)
            .First();
    }
}

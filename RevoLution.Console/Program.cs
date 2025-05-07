using RevoLution.Evolution;
using RevoLution.Hybrid;
using RevoLution.Neural;
using RevoLution.RL;
using RevoLution.Novelty;
using RevoLution.Environments;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RevoLution.Console;

class Program
{    
    static void Main(string[] args)
    {
        System.Console.WriteLine("RevoLution: Hybrid Neural Network Evolution with NEAT, Novelty Search, and RL");
        System.Console.WriteLine("==========================================================================");
        System.Console.WriteLine();

        // Use CartPole environment for simplicity and better performance
        RunCartPoleSimpleDemo();
    }
    
    static void RunCartPoleSimpleDemo()
    {
        System.Console.WriteLine("Running CartPole Simple Demo...");
        
        // Create environment
        var environment = new CartPoleEnvironment();
        
        // Create a simple neural network to show how it works
        var network = CreateSimpleNetwork(4, 1, 3);
        
        // Test the network with random weights
        System.Console.WriteLine("\nTesting network with random weights:");
        double totalReward = 0;
        int episodes = 3;
        
        for (int i = 0; i < episodes; i++)
        {
            var state = environment.Reset();
            double episodeReward = 0;
            bool done = false;
            int steps = 0;
            
            while (!done && steps < 200)
            {
                var outputs = network.FeedForward(state);
                var (nextState, reward, isDone) = environment.Step(outputs);
                
                state = nextState;
                episodeReward += reward;
                done = isDone;
                steps++;
            }
            
            System.Console.WriteLine($"Episode {i+1}: Reward = {episodeReward}, Steps = {steps}");
            totalReward += episodeReward;
        }
        
        System.Console.WriteLine($"Average reward: {totalReward / episodes}");
        
        // Train with basic reinforcement learning
        System.Console.WriteLine("\nTraining with reinforcement learning...");
        var rl = new ReinforcementLearner
        {
            LearningRate = 0.01,
            DiscountFactor = 0.99,
            ExplorationRate = 0.1,
            BatchSize = 16
        };
        
        // Define the environment function for RL
        Func<List<double>, List<double>, (List<double>, double, bool)> environmentFunction = (state, action) =>
        {
            // If state is null or empty, reset the environment
            if (state == null || state.Count == 0)
            {
                var initialState = environment.Reset();
                return (initialState, 0, false);
            }
                
            // Otherwise, take a step in the environment
            return environment.Step(action);
        };
        
        // Train for a few episodes
        rl.Train(network, environmentFunction, episodes: 10, maxStepsPerEpisode: 200);
        
        // Test the trained network
        System.Console.WriteLine("\nTesting network after reinforcement learning:");
        totalReward = 0;
        
        for (int i = 0; i < episodes; i++)
        {
            var state = environment.Reset();
            double episodeReward = 0;
            bool done = false;
            int steps = 0;
            
            while (!done && steps < 200)
            {
                var outputs = network.FeedForward(state);
                var (nextState, reward, isDone) = environment.Step(outputs);
                
                state = nextState;
                episodeReward += reward;
                done = isDone;
                steps++;
            }
            
            System.Console.WriteLine($"Episode {i+1}: Reward = {episodeReward}, Steps = {steps}");
            totalReward += episodeReward;
        }
        
        System.Console.WriteLine($"Average reward: {totalReward / episodes}");
        
        // Now demonstrate the hybrid approach with NEAT and novelty search
        System.Console.WriteLine("\nDemonstrating hybrid learning approach (simplified)...");
        var hybridLearner = new HybridLearner(
            new GeneticAlgorithm
            {
                PopulationSize = 10, // Smaller for quicker demo
                MutateWeightChance = 0.8,
                AddNodeChance = 0.03,
                AddConnectionChance = 0.05
            },
            new ReinforcementLearner
            {
                LearningRate = 0.01,
                DiscountFactor = 0.99,
                ExplorationRate = 0.1,
                BatchSize = 16
            },
            new NoveltySearch
            {
                K = 5,
                NoveltyThreshold = 0.5
            }
        )
        {
            NoveltyWeight = 0.4,
            TaskWeight = 0.6,
            TrainingEpisodes = 5, // Smaller for quicker demo
            EvaluationEpisodes = 2,
            GenerationsPerCycle = 2
        };
        
        // Initialize the population
        hybridLearner.Initialize(4, 1, 2);
        
        // Define evaluation function for NEAT
        Func<NeuralNetwork, (double, double[])> evaluateFunction = network =>
        {
            var state = environment.Reset();
            double totalReward = 0;
            bool done = false;
            int steps = 0;
            
            while (!done && steps < 100) // Shorter episodes for demo
            {
                var outputs = network.FeedForward(state);
                var (nextState, reward, isDone) = environment.Step(outputs);
                
                state = nextState;
                totalReward += reward;
                done = isDone;
                steps++;
            }
              // Get behavior characteristics to measure novelty
            var behavior = environment.GetBehaviorCharacteristics();
            return (totalReward, behavior);
        };
        
        // Run a single cycle of hybrid learning
        System.Console.WriteLine("Running a quick demonstration of hybrid learning...");
        hybridLearner.Learn(
            evaluateFunction,
            environmentFunction,
            cycles: 1
        );
        
        // Get and test the best network
        var bestNetwork = hybridLearner.GetBestSpecializedNetwork();
        if (bestNetwork != null)
        {
            System.Console.WriteLine("\nPerformance of best specialized network:");
            var state = environment.Reset();
            double bestReward = 0;
            bool done = false;
            int steps = 0;
            
            while (!done && steps < 200)
            {
                var outputs = bestNetwork.FeedForward(state);
                var (nextState, reward, isDone) = environment.Step(outputs);
                
                state = nextState;
                bestReward += reward;
                done = isDone;
                steps++;
            }
            
            System.Console.WriteLine($"Reward: {bestReward}, Steps: {steps}");
        }
        
        // Explain the hybrid approach
        System.Console.WriteLine("\nThe hybrid approach combines:");
        System.Console.WriteLine("1. NEAT: To evolve network architectures");
        System.Console.WriteLine("2. Novelty Search: To promote behavioral diversity");
        System.Console.WriteLine("3. RL: To fine-tune promising networks");
        System.Console.WriteLine("\nThis approach mirrors the parallel terraced scan concept from Copycat");
        System.Console.WriteLine("where the system explores both breadth (diversity of networks through");
        System.Console.WriteLine("NEAT and novelty search) and depth (specialization through RL).");
        System.Console.WriteLine("\nLike Copycat's architecture, this hybrid approach balances:");
        System.Console.WriteLine("- Exploration: Finding novel network architectures and behaviors");
        System.Console.WriteLine("- Exploitation: Optimizing promising solutions with RL");
    }
    
    static NeuralNetwork CreateSimpleNetwork(int inputSize, int outputSize, int hiddenNodes)
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
        
        // Create random connections
        var random = new Random();
        
        // Connect inputs to hidden
        for (int i = 0; i < inputSize; i++)
        {
            for (int h = inputSize; h < inputSize + hiddenNodes; h++)
            {
                connections.Add(new Connection(i, h, RandomWeight(random), innovationNumber++));
            }
        }
        
        // Connect hidden to outputs
        for (int h = inputSize; h < inputSize + hiddenNodes; h++)
        {
            for (int o = inputSize + hiddenNodes; o < inputSize + hiddenNodes + outputSize; o++)
            {
                connections.Add(new Connection(h, o, RandomWeight(random), innovationNumber++));
            }
        }
        
        return new NeuralNetwork(nodes, connections);
    }
    
    static double RandomWeight(Random random)
    {
        return (random.NextDouble() * 4) - 2; // Between -2 and 2
    }
}

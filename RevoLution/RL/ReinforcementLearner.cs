using System;
using System.Collections.Generic;
using System.Linq;

namespace RevoLution.RL;

using RevoLution.Neural;

public class ReinforcementLearner
{
    private readonly Random _random = new();
    
    // RL Hyperparameters
    public double LearningRate { get; set; } = 0.01;
    public double DiscountFactor { get; set; } = 0.99;
    public double ExplorationRate { get; set; } = 0.1;
    public double ExplorationDecay { get; set; } = 0.995;
    public double MinExplorationRate { get; set; } = 0.01;
    public int BatchSize { get; set; } = 32;
    public int ReplayBufferCapacity { get; set; } = 10000;
    
    // Experience replay buffer
    private readonly Queue<Experience> _replayBuffer = new();
    
    // Training metrics
    public List<double> RewardHistory { get; } = new();
    public List<double> LossHistory { get; } = new();

    public ReinforcementLearner() { }

    public void Train(NeuralNetwork network, Func<List<double>, List<double>, (List<double>, double, bool)> environment, 
                      int episodes = 1000, int maxStepsPerEpisode = 200)
    {
        for (int episode = 0; episode < episodes; episode++)
        {            // Reset environment
            var initialState = new List<double>();
            var state = environment(initialState, initialState).Item1;
            double totalReward = 0;
            
            for (int step = 0; step < maxStepsPerEpisode; step++)
            {
                // Choose action (epsilon-greedy)
                List<double> action;
                if (_random.NextDouble() < ExplorationRate)
                {
                    // Random action
                    action = Enumerable.Range(0, network.FeedForward(state).Count)
                              .Select(_ => _random.NextDouble())
                              .ToList();
                }
                else
                {
                    // Use network to select action
                    action = network.FeedForward(state);
                }
                
                // Take action in environment
                var (nextState, reward, done) = environment(state, action);
                
                // Store experience in replay buffer
                _replayBuffer.Enqueue(new Experience(state, action, reward, nextState, done));
                
                // Ensure buffer doesn't exceed capacity
                if (_replayBuffer.Count > ReplayBufferCapacity)
                {
                    _replayBuffer.Dequeue();
                }
                
                // Update state and accumulate reward
                state = nextState;
                totalReward += reward;
                
                // Perform learning if we have enough experiences
                if (_replayBuffer.Count >= BatchSize)
                {
                    double loss = LearnFromExperiences(network);
                    LossHistory.Add(loss);
                }
                
                // Break if episode is done
                if (done)
                    break;
            }
            
            // Decay exploration rate
            ExplorationRate = Math.Max(MinExplorationRate, ExplorationRate * ExplorationDecay);
            
            // Record reward for this episode
            RewardHistory.Add(totalReward);
        }
    }

    private double LearnFromExperiences(NeuralNetwork network)
    {
        // Sample random batch from replay buffer
        var batch = _replayBuffer.OrderBy(_ => _random.Next()).Take(BatchSize).ToList();
        double totalLoss = 0;
        
        foreach (var experience in batch)
        {
            // Get current Q values
            var currentQValues = network.FeedForward(experience.State);
            
            // Create target Q values (same as current to start)
            var targetQValues = new List<double>(currentQValues);
            
            // Calculate target value (Q-learning)
            if (experience.Done)
            {
                // If terminal state, target is just the reward
                for (int i = 0; i < targetQValues.Count; i++)
                {
                    if (i == GetMaxIndex(experience.Action))
                    {
                        targetQValues[i] = experience.Reward;
                    }
                }
            }
            else
            {                // Get future Q values from next state
                var nextQValues = network.FeedForward(experience.NextState);
                
                // Make sure we have values before taking the max
                double maxNextQ = 0;
                if (nextQValues.Count > 0)
                {
                    maxNextQ = nextQValues.Max();
                }
                else
                {
                    Console.WriteLine("Warning: Network returned empty output in RL update");
                }
                
                // Update target for the action that was taken
                int actionIndex = GetMaxIndex(experience.Action);
                targetQValues[actionIndex] = experience.Reward + DiscountFactor * maxNextQ;
            }
            
            // Backpropagate with the target Q values
            network.Backpropagate(experience.State, targetQValues, LearningRate);
            
            // Calculate loss (MSE)
            double loss = network.CalculateLoss(currentQValues, targetQValues);
            totalLoss += loss;
        }
        
        return totalLoss / BatchSize;
    }    private int GetMaxIndex(List<double> values)
    {
        if (values == null || values.Count == 0)
        {
            Console.WriteLine("Warning: Empty values list in GetMaxIndex");
            return 0;
        }
        
        int maxIndex = 0;
        double maxValue = values[0];
        
        for (int i = 1; i < values.Count; i++)
        {
            if (values[i] > maxValue)
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
}

public class Experience
{
    public List<double> State { get; }
    public List<double> Action { get; }
    public double Reward { get; }
    public List<double> NextState { get; }
    public bool Done { get; }

    public Experience(List<double> state, List<double> action, double reward, List<double> nextState, bool done)
    {
        State = state;
        Action = action;
        Reward = reward;
        NextState = nextState;
        Done = done;
    }
}

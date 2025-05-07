using System;
using System.Collections.Generic;
using System.Linq;

namespace RevoLution.Environments;

/// <summary>
/// Base interface for environments that can be used with the hybrid learning system.
/// </summary>
public interface IEnvironment
{
    /// <summary>
    /// Resets the environment to its initial state.
    /// </summary>
    /// <returns>The initial state observation.</returns>
    List<double> Reset();
    
    /// <summary>
    /// Takes an action in the environment.
    /// </summary>
    /// <param name="action">The action to take.</param>
    /// <returns>The next state, reward, and whether the episode is done.</returns>
    (List<double>, double, bool) Step(List<double> action);
    
    /// <summary>
    /// Gets a description of the current state that can be used for behavior characterization.
    /// </summary>
    /// <returns>An array representing the behavior characteristics.</returns>
    double[] GetBehaviorCharacteristics();
}

/// <summary>
/// A basic cart-pole environment for testing the learning algorithms.
/// </summary>
public class CartPoleEnvironment : IEnvironment
{
    private const double Gravity = 9.8;
    private const double MassCart = 1.0;
    private const double MassPole = 0.1;
    private const double TotalMass = MassCart + MassPole;
    private const double Length = 0.5; // half the pole's length
    private const double PoleMassLength = MassPole * Length;
    private const double ForceMag = 10.0;
    private const double Tau = 0.02; // seconds between state updates
    
    // Angle at which to fail the episode
    private const double ThetaThresholdRadians = 12 * 2 * Math.PI / 360;
    private const double XThreshold = 2.4;
    
    // Current state: [position, velocity, angle, angular velocity]
    private double[] _state = new double[4];
    private readonly Random _random = new();
    private int _steps = 0;
    private readonly int _maxSteps = 500;

    public List<double> Reset()
    {
        _state[0] = _random.NextDouble() * 0.1 - 0.05; // position
        _state[1] = _random.NextDouble() * 0.1 - 0.05; // velocity
        _state[2] = _random.NextDouble() * 0.1 - 0.05; // angle
        _state[3] = _random.NextDouble() * 0.1 - 0.05; // angular velocity
        
        _steps = 0;
        
        return _state.ToList();
    }    public (List<double>, double, bool) Step(List<double> action)
    {
        // Extract the current state
        double x = _state[0];
        double xDot = _state[1];
        double theta = _state[2];
        double thetaDot = _state[3];
        
        // Convert action to force (-1 or 1), with safety check for empty actions
        double force;
        if (action == null || action.Count == 0)
        {
            // Default to random action if no action is provided
            force = _random.NextDouble() > 0.5 ? ForceMag : -ForceMag;
            Console.WriteLine("Warning: Empty action received in CartPoleEnvironment. Using random action.");
        }
        else
        {
            force = action[0] > 0.5 ? ForceMag : -ForceMag;
        }
        
        // Calculate dynamics
        double cosTheta = Math.Cos(theta);
        double sinTheta = Math.Sin(theta);
        
        // Calculate temporary terms to avoid division by zero
        double temp = (force + PoleMassLength * thetaDot * thetaDot * sinTheta) / TotalMass;
        double thetaAcc = (Gravity * sinTheta - cosTheta * temp) / 
            (Length * (4.0/3.0 - MassPole * cosTheta * cosTheta / TotalMass));
        double xAcc = temp - PoleMassLength * thetaAcc * cosTheta / TotalMass;
        
        // Update state using Euler integration
        x += Tau * xDot;
        xDot += Tau * xAcc;
        theta += Tau * thetaDot;
        thetaDot += Tau * thetaAcc;
        
        // Store the new state
        _state[0] = x;
        _state[1] = xDot;
        _state[2] = theta;
        _state[3] = thetaDot;
        
        _steps++;
        
        // Check if we're done
        bool isDone = 
            x < -XThreshold || 
            x > XThreshold || 
            theta < -ThetaThresholdRadians || 
            theta > ThetaThresholdRadians ||
            _steps >= _maxSteps;
        
        // Calculate reward
        double reward = 1.0;
        if (isDone && _steps < _maxSteps)
        {
            reward = 0.0; // Penalty for failure
        }
        
        return (_state.ToList(), reward, isDone);
    }

    public double[] GetBehaviorCharacteristics()
    {
        // For a simple behavior characterization, we'll use:
        // - Final position
        // - Maximum angle reached
        // - Average velocity
        // - Average angular velocity
        
        // Compute average velocity and angular velocity over time
        double avgXVel = _state[1];
        double avgAngVel = _state[3];
        
        // Return behavior characteristics
        return new double[] 
        {
            _state[0],          // Final position
            _state[2],          // Final angle
            avgXVel,            // Average velocity
            avgAngVel           // Average angular velocity
        };
    }
}

/// <summary>
/// A simple maze environment to test exploration and behavioral diversity.
/// </summary>
public class MazeEnvironment : IEnvironment
{
    // Maze parameters
    private readonly int _width;
    private readonly int _height;
    private readonly int[,] _maze;
    
    // Agent state
    private int _agentX;
    private int _agentY;
    private readonly int _startX;
    private readonly int _startY;
    private readonly int _goalX;
    private readonly int _goalY;
    
    // Behavior tracking
    private readonly List<(int, int)> _visitedPositions = new();
    private int _steps = 0;
    private readonly int _maxSteps = 200;
    
    // Constants
    private const int Wall = 1;
    private const int Empty = 0;
    private const int Start = 2;
    private const int Goal = 3;

    public MazeEnvironment(int width = 10, int height = 10)
    {
        _width = width;
        _height = height;
        
        // Generate a simple maze
        _maze = GenerateMaze(width, height);
        
        // Find start and goal positions
        (_startX, _startY) = FindPosition(Start);
        (_goalX, _goalY) = FindPosition(Goal);
        
        _agentX = _startX;
        _agentY = _startY;
    }
    
    private int[,] GenerateMaze(int width, int height)
    {
        int[,] maze = new int[width, height];
        Random random = new Random();
        
        // Fill with walls
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                maze[x, y] = Wall;
            }
        }
        
        // Generate a simple random maze using a modified depth-first search
        Stack<(int, int)> stack = new Stack<(int, int)>();
        stack.Push((1, 1));
        maze[1, 1] = Empty;
        
        while (stack.Count > 0)
        {
            var (x, y) = stack.Peek();
            var neighbors = GetUnvisitedNeighbors(maze, x, y, width, height);
            
            if (neighbors.Count > 0)
            {
                var next = neighbors[random.Next(neighbors.Count)];
                maze[next.Item1, next.Item2] = Empty;
                
                // Clear the wall between the current cell and the next cell
                maze[(x + next.Item1) / 2, (y + next.Item2) / 2] = Empty;
                
                stack.Push(next);
            }
            else
            {
                stack.Pop();
            }
        }
        
        // Place start and goal
        maze[1, 1] = Start;
        maze[width - 2, height - 2] = Goal;
        
        return maze;
    }

    private List<(int, int)> GetUnvisitedNeighbors(int[,] maze, int x, int y, int width, int height)
    {
        var neighbors = new List<(int, int)>();
        
        // Check all four directions
        int[][] directions = new int[][]
        {
            new int[] { 0, -2 }, // Up
            new int[] { 2, 0 },  // Right
            new int[] { 0, 2 },  // Down
            new int[] { -2, 0 }  // Left
        };
        
        foreach (var dir in directions)
        {
            int nx = x + dir[0];
            int ny = y + dir[1];
            
            if (nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1 && maze[nx, ny] == Wall)
            {
                neighbors.Add((nx, ny));
            }
        }
        
        return neighbors;
    }

    private (int, int) FindPosition(int target)
    {
        for (int x = 0; x < _width; x++)
        {
            for (int y = 0; y < _height; y++)
            {
                if (_maze[x, y] == target)
                {
                    return (x, y);
                }
            }
        }
        
        throw new InvalidOperationException($"Position {target} not found in maze");
    }

    public List<double> Reset()
    {
        _agentX = _startX;
        _agentY = _startY;
        _visitedPositions.Clear();
        _visitedPositions.Add((_agentX, _agentY));
        _steps = 0;
        
        return GetObservation();
    }

    private List<double> GetObservation()
    {
        // Create simple observation: [agent X, agent Y, goal X, goal Y, 
        //                            wall up, wall right, wall down, wall left]
        var obs = new List<double>
        {
            _agentX / (double)_width,
            _agentY / (double)_height,
            _goalX / (double)_width,
            _goalY / (double)_height
        };
        
        // Check for walls in each direction
        obs.Add(IsWall(_agentX, _agentY - 1) ? 1.0 : 0.0); // Up
        obs.Add(IsWall(_agentX + 1, _agentY) ? 1.0 : 0.0); // Right
        obs.Add(IsWall(_agentX, _agentY + 1) ? 1.0 : 0.0); // Down
        obs.Add(IsWall(_agentX - 1, _agentY) ? 1.0 : 0.0); // Left
        
        return obs;
    }

    private bool IsWall(int x, int y)
    {
        if (x < 0 || x >= _width || y < 0 || y >= _height)
            return true; // Out of bounds is a wall
            
        return _maze[x, y] == Wall;
    }

    public (List<double>, double, bool) Step(List<double> action)
    {
        // Interpret action (assumes one-hot or continuous values)
        int actionIdx = 0;
        if (action.Count > 1)
        {
            // One-hot action
            actionIdx = action.IndexOf(action.Max());
        }
        else
        {
            // Continuous action
            actionIdx = (int)Math.Floor(action[0] * 4) % 4;
        }
        
        // Calculate new position
        int newX = _agentX;
        int newY = _agentY;
        
        switch (actionIdx)
        {
            case 0: newY--; break; // Up
            case 1: newX++; break; // Right
            case 2: newY++; break; // Down
            case 3: newX--; break; // Left
        }
        
        // Check if the new position is valid
        if (!IsWall(newX, newY))
        {
            _agentX = newX;
            _agentY = newY;
        }
        
        // Track visited positions
        _visitedPositions.Add((_agentX, _agentY));
        
        // Increment step counter
        _steps++;
        
        // Check if reached goal
        bool reachedGoal = _agentX == _goalX && _agentY == _goalY;
        bool timeout = _steps >= _maxSteps;
        
        // Calculate reward
        double reward = 0.0;
        
        if (reachedGoal)
        {
            reward = 1.0;
        }
        else
        {
            // Small penalty for each step to encourage efficiency
            reward = -0.01;
            
            // Small bonus for getting closer to the goal
            double prevDistance = Distance(_startX, _startY, _goalX, _goalY);
            double newDistance = Distance(_agentX, _agentY, _goalX, _goalY);
            reward += 0.02 * (prevDistance - newDistance);
        }
        
        return (GetObservation(), reward, reachedGoal || timeout);
    }

    private double Distance(int x1, int y1, int x2, int y2)
    {
        return Math.Sqrt(Math.Pow(x2 - x1, 2) + Math.Pow(y2 - y1, 2));
    }

    public double[] GetBehaviorCharacteristics()
    {
        // For maze behavioral characteristics, we'll use:
        // - Number of unique cells visited
        // - Final distance to goal
        // - Overall path coverage (unique cells / total cells)
        // - Directional bias (how much the agent preferred certain directions)
        
        int uniqueVisited = _visitedPositions.Distinct().Count();
        double finalDistance = Distance(_agentX, _agentY, _goalX, _goalY);
        double coverage = (double)uniqueVisited / (_width * _height);
        
        // Calculate directional bias (up/down vs left/right)
        int horizontalMoves = 0;
        int verticalMoves = 0;
        
        for (int i = 1; i < _visitedPositions.Count; i++)
        {
            var prev = _visitedPositions[i - 1];
            var curr = _visitedPositions[i];
            
            if (prev.Item1 != curr.Item1)
                horizontalMoves++;
                
            if (prev.Item2 != curr.Item2)
                verticalMoves++;
        }
        
        double directionalBias = 0.5;
        if (horizontalMoves + verticalMoves > 0)
        {
            directionalBias = (double)horizontalMoves / (horizontalMoves + verticalMoves);
        }
        
        return new double[]
        {
            uniqueVisited / (double)(_width * _height),
            finalDistance / Math.Sqrt(_width * _width + _height * _height),
            coverage,
            directionalBias
        };
    }
}

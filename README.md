# RevoLution: Hybrid Neural Learning System

A C# implementation combining evolutionary neural networks (NEAT), reinforcement learning, and novelty search into a hybrid learning system. This project is inspired by Douglas Hofstadter and Melanie Mitchell's Copycat project, creating a cognitive architecture that blends different learning approaches.

## Features

- **NEAT (NeuroEvolution of Augmenting Topologies)**: Networks can evolve their architecture through genetic algorithms
- **Reinforcement Learning**: Networks can specialize through backpropagation-based reinforcement learning
- **Novelty Search**: Promotes diversity in the population through behavioral novelty
- **Hybrid Approach**: Combines all three approaches for robust learning and problem-solving

## Components

- **Neural**: Core neural network implementation
- **Evolution**: NEAT-based genetic algorithm for evolving network topologies
- **RL**: Reinforcement learning implementation
- **Hybrid**: Integration of all learning approaches
- **Environments**: Test environments including CartPole

## Demo

The project includes a simple CartPole demo that showcases the hybrid learning approach, demonstrating how the system can:
- Evolve network structures
- Apply reinforcement learning to improve performance
- Maintain diversity through novelty search

## Getting Started

1. Clone the repository
2. Build the solution using Visual Studio or dotnet CLI
3. Run the RevoLution.Console project to see the demo

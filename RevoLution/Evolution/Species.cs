using System;
using System.Collections.Generic;

namespace RevoLution.Evolution;

using RevoLution.Neural;

public class Species
{
    public int Id { get; }
    public List<NeuralNetwork> Networks { get; set; } = new();
    public NeuralNetwork Representative { get; set; }
    public double BestFitness { get; set; }
    public int StagnationCount { get; set; }
    public int Generation { get; set; }

    public Species(int id, NeuralNetwork representative)
    {
        Id = id;
        Representative = representative;
        Networks.Add(representative);
        representative.SpeciesId = id;
    }

    public void CalculateAdjustedFitness()
    {
        foreach (var network in Networks)
        {
            network.AdjustedFitness = network.Fitness / Networks.Count;
        }
    }

    public void UpdateRepresentative()
    {
        if (Networks.Count > 0)
        {
            Representative = Networks[new Random().Next(Networks.Count)];
        }
    }
}

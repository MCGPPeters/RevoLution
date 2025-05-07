using System;
using System.Collections.Generic;
using System.Linq;

namespace RevoLution.Novelty;

using RevoLution.Neural;

public class NoveltySearch
{
    private readonly List<double[]> _noveltyArchive = new();
    private readonly Random _random = new();
    
    // Configuration
    public int ArchiveSize { get; set; } = 1000;
    public int K { get; set; } = 15; // k-nearest neighbors
    public double NoveltyThreshold { get; set; } = 0.5;
    public double AddToArchiveChance { get; set; } = 0.05;
    
    public double CalculateNovelty(double[] behavior, List<double[]> population)
    {
        var neighbors = new List<double[]>();
        
        // Add archive points
        neighbors.AddRange(_noveltyArchive);
        
        // Add current population behaviors (excluding the current one)
        neighbors.AddRange(population.Where(b => b != behavior));
        
        if (neighbors.Count == 0)
        {
            // If no neighbors, consider maximum novelty
            return double.MaxValue;
        }
        
        // Calculate distances to all neighbors
        var distances = neighbors.Select(n => EuclideanDistance(behavior, n)).ToList();
        
        // Sort distances
        distances.Sort();
        
        // Take average of k nearest neighbors
        int kNN = Math.Min(K, distances.Count);
        double avgDistance = 0;
        
        for (int i = 0; i < kNN; i++)
        {
            avgDistance += distances[i];
        }
        
        avgDistance /= kNN;
        
        // Potentially add to archive
        if (avgDistance > NoveltyThreshold || _random.NextDouble() < AddToArchiveChance)
        {
            AddToArchive(behavior);
        }
        
        return avgDistance;
    }

    private void AddToArchive(double[] behavior)
    {
        // Add behavior to archive
        _noveltyArchive.Add((double[])behavior.Clone());
        
        // Trim archive if it exceeds maximum size
        while (_noveltyArchive.Count > ArchiveSize)
        {
            // Remove random entry
            int indexToRemove = _random.Next(_noveltyArchive.Count);
            _noveltyArchive.RemoveAt(indexToRemove);
        }
    }

    private double EuclideanDistance(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same dimension");
        
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        
        return Math.Sqrt(sum);
    }
}

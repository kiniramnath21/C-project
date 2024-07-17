using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

public class Transaction
{
    [LoadColumn(0)]
    public float Amount { get; set; }

    [LoadColumn(1)]
    public float Time { get; set; }

    [LoadColumn(2)]
    public float V1 { get; set; }

    [LoadColumn(30)] // Assuming label column is at index 30
    [ColumnName("Label")]
    public bool IsFraudulent { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        // Set up MLContext
        MLContext mlContext = new MLContext(seed: 1);

        // Load data
        IDataView dataView = mlContext.Data.LoadFromTextFile<Transaction>("fraudTest.csv", separatorChar: ',');

        // Data preprocessing (if needed)
        var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", "Amount", "Time", "V1")
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));

        // Split data into train and test sets
        var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

        // Define your machine learning pipeline
        var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
        var trainingPipeline = dataProcessPipeline.Append(trainer);

        // Train the model
        var trainedModel = trainingPipeline.Fit(trainTestData.TrainSet);

        // Evaluate the model
        var predictions = trainedModel.Transform(trainTestData.TestSet);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

        // Output evaluation metrics
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"Precision: {metrics.Precision:P2}");
        Console.WriteLine($"Recall: {metrics.Recall:P2}");
        Console.WriteLine($"F1-score: {metrics.F1Score:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");

        // Save the model
        mlContext.Model.Save(trainedModel, trainTestData.TrainSet.Schema, "credit-card-fraud-model.zip");
        Console.WriteLine("Model saved to 'credit-card-fraud-model.zip'.");
    }
}

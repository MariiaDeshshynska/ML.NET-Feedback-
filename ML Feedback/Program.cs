using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;

namespace ML_Feedback
{
    class Program
    {
        static List<FeedBackTrainingData> trainingData = new List<FeedBackTrainingData>();
        static List<FeedBackTrainingData> testData = new List<FeedBackTrainingData>();
        static void LoadTrainingData()
        {
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is good",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice and smooth",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "shitty horrible",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is bad",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "as usual amazing",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "fully disapointed",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "worse than ever",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "well ok ok",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "wow that's great",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "very confusing",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "very unpleasant",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "holly shit",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "no comments",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "such an awful experience",
                IsGood = false
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "such a service",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "so nice",
                IsGood = true
            });
            trainingData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice and pleasant",
                IsGood = true
            });
        }
        static void LoadTestData()
        {
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "good",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "horrible",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "amazing",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "disapointed",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "worse",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "well",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "great",
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "confusing",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "unpleasant",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "shit",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "no comments",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "awful",
                IsGood = false
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "such a service",
                IsGood = true
            });
        }
        static void Main(string[] args)
        {
            LoadTrainingData();

            var mlContext = new MLContext();

            var dataView = mlContext.CreateStreamingDataView<FeedBackTrainingData>(trainingData);

            var pipeline = mlContext.Transforms.Text.FeaturizeText("FeedBackText", "Features")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 1)) ;
            var model = pipeline.Fit(dataView);

            LoadTestData();

            var dataView1 = mlContext.CreateStreamingDataView<FeedBackTrainingData>(testData);
            var predictions = model.Transform(dataView1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine(metrics.Accuracy);
            Console.WriteLine("Please leave your feedback");
            string feedbackstring = Console.ReadLine().ToString();

            var predictionFunction = model.MakePredictionFunction<FeedBackTrainingData, FeedBackPrediction>(mlContext);

            var feedbackinput = new FeedBackTrainingData();
            feedbackinput.FeedBackText = feedbackstring;
            var feedbackpredicted = predictionFunction.Predict(feedbackinput);
            Console.WriteLine("Predicted : " + feedbackpredicted.IsGood);
            Console.ReadLine();

        }
    }
}

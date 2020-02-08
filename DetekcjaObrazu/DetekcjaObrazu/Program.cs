using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace DetekcjaObrazu
{
    public class Program
    {
        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args">The command line arguments></param>
        static void Main(string[] args)
        {
            // context AI Microsoftu
            var mlContext = new MLContext();

            // ladowanie dolaczonych plikow w formacie tsv
            var data = mlContext.Data.LoadFromTextFile<ImageDataModel>("input/tagi.tsv", hasHeader: true);

            // Ladowanie systemu pipeline
            var pipeline = mlContext.Transforms
                .LoadImages(
                    outputColumnName: "input",
                    imageFolder: "Referencje",
                    inputColumnName: nameof(ImageDataModel.ImagePath))

                // step 2: zmiana wielkosci plikow 224x224
                .Append(mlContext.Transforms.ResizeImages(
                    outputColumnName: "input",
                    imageWidth: 224,
                    imageHeight: 224,
                    inputColumnName: "input"))

                // step 3: Wyciaganie pixeli z pliku TF
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "input",
                    interleavePixelColors: true,
                    offsetImage: 117))

                // step 4: lLaduje flow programu
                .Append(mlContext.Model.LoadTensorFlowModel("input/graph_danych.pb")

                // step 5: skaluje obraz do pliku TF
                .ScoreTensorFlowModel(
                    outputColumnNames: new[] { "softmax2" },
                    inputColumnNames: new[] { "input" },
                    addBatchDimensionInput: true));

            // Uczy model
            Console.WriteLine("Startuje uczenie modelu !");
            var model = pipeline.Fit(data);
            Console.WriteLine("Trenowanie modelu zakonczone!");


            var engine = mlContext.Model.CreatePredictionEngine<ImageDataModel, ImageContainerPrediction>(model);

            // Laduje labele obrazow
            var labels = File.ReadAllLines("input/labelGrafowStrings.txt");

            // predict co jest w obrazie
            Console.WriteLine("Predykcja obrazu....");
            var images = ImageDataModel.ReadFromCsv("input/tagi.tsv");
            foreach (var image in images)
            {
                Console.Write($"  [{image.ImagePath}]: ");
                var prediction = engine.Predict(image).PredictedLabele;

                // Znajdz nalepsza predykcje
                var i = 0;
                var best = (from p in prediction
                            select new { Index = i++, Prediction = p }).OrderByDescending(p => p.Prediction).First();
                var predictedLabel = labels[best.Index];

                // Pokaz labele
                Console.WriteLine($"{predictedLabel} {(predictedLabel != image.Label ? "BLAD!" : "")}");
            }
        }
    }
}

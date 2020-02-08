using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace DetekcjaObrazu
{
    class ImageDataModel
    {
        [LoadColumn(0)] public string ImagePath;
        [LoadColumn(1)] public string Label;

        /// <summary>
        /// Laduje pliki CSV jako sekwencje
        /// </summary>
        /// <param name="file">Plik tsv</param>
        /// <returns>Zwraca sekwencje plikow ladowanych obrazow</returns>
        public static IEnumerable<ImageDataModel> ReadFromCsv(string file)
        {
            return File.ReadAllLines(file)
                .Select(x => x.Split('\t'))
                .Select(x => new ImageDataModel
                {
                    ImagePath = x[0],
                    Label = x[1]
                });
        }
    }
}

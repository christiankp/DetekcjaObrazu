using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace DetekcjaObrazu
{
    class ImageContainerPrediction
    {
        [ColumnName("softmax2")]
        public float[] PredictedLabele;
    }
}

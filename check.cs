using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;

class Program
{
    static void Main()
    {
        var session = new InferenceSession(@"c:\D\Codes\SpatialVisionService\Models\FineTunedPotHole.onnx");
        foreach(var meta in session.OutputMetadata)
        {
            Console.WriteLine(meta.Key + ": " + string.Join(",", meta.Value.Dimensions));
        }
    }
}

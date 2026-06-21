using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SpatialVisionService.Services
{
    public class DepthEstimator
    {
        private const int TargetSize = 384;
        private readonly float[] _mean = [0.485f, 0.456f, 0.406f];
        private readonly float[] _std = [0.229f, 0.224f, 0.225f];

        public void Process(string modelPath, string inputPath, string outputPath)
        {
            Console.WriteLine("--- Phase 1: Depth Mapping ---");

            using var session = new InferenceSession(modelPath);
            using var image = Image.Load<Rgb24>(inputPath);

            image.Mutate(x => x.Resize(TargetSize, TargetSize));

            var inputTensor = Preprocess(image);
            var inputName = session.InputMetadata.Keys.First();

            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

            using var results = session.Run(inputs);
            var depthTensor = results.First().AsTensor<float>();

            PostProcessAndSave(depthTensor, outputPath);
        }

        private DenseTensor<float> Preprocess(Image<Rgb24> image)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });

            for (int y = 0; y < TargetSize; y++)
            {
                for (int x = 0; x < TargetSize; x++)
                {
                    Rgb24 pixel = image[x, y];
                    tensor[0, 0, y, x] = ((pixel.R / 255f) - _mean[0]) / _std[0];
                    tensor[0, 1, y, x] = ((pixel.G / 255f) - _mean[1]) / _std[1];
                    tensor[0, 2, y, x] = ((pixel.B / 255f) - _mean[2]) / _std[2];
                }
            }
            return tensor;
        }
        private void PostProcessAndSave(Tensor<float> depthTensor, string outputPath)
        {
            float[] data = depthTensor.ToArray();

            // 1. Sort the data to find the percentiles
            float[] sortedData = (float[])data.Clone();
            Array.Sort(sortedData);

            // 2. Pick the 2nd percentile and 98th percentile as our bounds
            int lowerIndex = (int)(sortedData.Length * 0.02);
            int upperIndex = (int)(sortedData.Length * 0.98);

            float minClip = sortedData[lowerIndex];
            float maxClip = sortedData[upperIndex];
            float range = maxClip - minClip;

            int height = (int)depthTensor.Dimensions[depthTensor.Dimensions.Length - 2];
            int width = (int)depthTensor.Dimensions[depthTensor.Dimensions.Length - 1];

            using var outputImage = new Image<L8>(width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float val = data[(y * width) + x];

                    // Clamp values to our clip range
                    float clamped = Math.Max(minClip, Math.Min(maxClip, val));

                    // Normalize based on the clipped range
                    float normalized = range > 0 ? (clamped - minClip) / range : 0f;

                    byte gray = (byte)(normalized * 255);
                    outputImage[x, y] = new L8(gray);
                }
            }

            outputImage.SaveAsPng(outputPath);
            Console.WriteLine($"Depth map saved with contrast stretch. Range used: {minClip:F2} to {maxClip:F2}");
        }
        //private void PostProcessAndSave(Tensor<float> depthTensor, string outputPath)
        //{
        //    // 1. Get raw data
        //    float[] data = depthTensor.ToArray();

        //    // 2. Find absolute Min/Max to fit the scale correctly
        //    float min = data.Min();
        //    float max = data.Max();
        //    float range = max - min;

        //    // Get dimensions from tensor (e.g., 384)
        //    int height = (int)depthTensor.Dimensions[depthTensor.Dimensions.Length - 2];
        //    int width = (int)depthTensor.Dimensions[depthTensor.Dimensions.Length - 1];

        //    using var outputImage = new Image<L8>(width, height);

        //    for (int y = 0; y < height; y++)
        //    {
        //        for (int x = 0; x < width; x++)
        //        {
        //            // Simple mapping: (value - min) / range
        //            float val = data[(y * width) + x];
        //            float normalized = range > 0 ? (val - min) / range : 0f;

        //            // Map 0.0-1.0 to 0-255
        //            byte gray = (byte)(normalized * 255);

        //            outputImage[x, y] = new L8(gray);
        //        }
        //    }

        //    outputImage.SaveAsPng(outputPath);
        //    Console.WriteLine($"Depth map saved to: {outputPath} (Range: {min:F2} to {max:F2})");
        //}
    }
}

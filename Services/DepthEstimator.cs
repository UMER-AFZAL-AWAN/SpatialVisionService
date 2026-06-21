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
            float minDepth = float.MaxValue;
            float maxDepth = float.MinValue;

            foreach (float val in depthTensor)
            {
                if (val < minDepth) minDepth = val;
                if (val > maxDepth) maxDepth = val;
            }

            using var outputImage = new Image<L8>(TargetSize, TargetSize);
            float depthRange = maxDepth - minDepth;

            for (int y = 0; y < TargetSize; y++)
            {
                for (int x = 0; x < TargetSize; x++)
                {
                    float rawDepth = depthTensor[0, y, x];
                    byte grayValue = (byte)(((rawDepth - minDepth) / depthRange) * 255);
                    outputImage[x, y] = new L8(grayValue);
                }
            }

            outputImage.SaveAsPng(outputPath);
            Console.WriteLine($"Depth map saved to: {outputPath}");
        }
    }
}

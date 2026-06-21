using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SpatialVisionService.Services
{
    public class ObjectDetector
    {
        private const int TargetSize = 640;
        private const float ConfidenceThreshold = 0.60f;

        public void Process(string modelPath, string inputPath, string outputPath)
        {
            Console.WriteLine("\n--- Phase 2: Object Detection ---");

            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"Warning: Could not find YOLO model at {modelPath}. Skipping.");
                return;
            }

            using var session = new InferenceSession(modelPath);
            using var originalImage = Image.Load<Rgb24>(inputPath);
            using var resizedImage = originalImage.Clone(x => x.Resize(TargetSize, TargetSize));

            var yoloTensor = Preprocess(resizedImage);
            string inputName = session.InputMetadata.Keys.First();

            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, yoloTensor)
        };

            using var results = session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            ParseAndDrawBoxes(outputTensor, originalImage, originalImage.Width, originalImage.Height);

            originalImage.SaveAsJpeg(outputPath);
            Console.WriteLine($"Bounding boxes saved to: {outputPath}");
        }

        private DenseTensor<float> Preprocess(Image<Rgb24> image)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });

            for (int y = 0; y < TargetSize; y++)
            {
                for (int x = 0; x < TargetSize; x++)
                {
                    Rgb24 pixel = image[x, y];
                    tensor[0, 0, y, x] = pixel.R / 255f;
                    tensor[0, 1, y, x] = pixel.G / 255f;
                    tensor[0, 2, y, x] = pixel.B / 255f;
                }
            }
            return tensor;
        }

        private void ParseAndDrawBoxes(Tensor<float> output, Image<Rgb24> targetImage, int origWidth, int origHeight)
        {
            // Handle native NMS-Free tensor layout: [Batch, BoxCount, 6]
            if (output.Dimensions.Length != 3 || output.Dimensions[2] != 6)
            {
                Console.WriteLine($"YOLO model returned unexpected shape: [{string.Join(", ", output.Dimensions.ToString())}].");
                return;
            }

            int numBoxes = output.Dimensions[1];
            float scaleX = (float)origWidth / TargetSize;
            float scaleY = (float)origHeight / TargetSize;

            for (int i = 0; i < numBoxes; i++)
            {
                float confidence = output[0, i, 4];
                if (confidence < ConfidenceThreshold) continue;

                // YOLO26 natively outputs [x1, y1, x2, y2, confidence, class_id]
                float x1 = output[0, i, 0];
                float y1 = output[0, i, 1];
                float x2 = output[0, i, 2];
                float y2 = output[0, i, 3];

                // Re-scale coordinate plane bounds back to native resolution targets
                float xMin = x1 * scaleX;
                float yMin = y1 * scaleY;
                float boxWidth = (x2 - x1) * scaleX;
                float boxHeight = (y2 - y1) * scaleY;

                var rect = new RectangleF(xMin, yMin, boxWidth, boxHeight);
                targetImage.Mutate(ctx => ctx.Draw(Color.LimeGreen, 3f, rect));
            }
        }
    }
}

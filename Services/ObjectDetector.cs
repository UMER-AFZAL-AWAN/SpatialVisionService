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
        private const float ConfidenceThreshold = 0.40f;
        public List<DetectionResult> Detect(string modelPath, string inputPath, out int originalWidth, out int originalHeight)
        {
            var detections = new List<DetectionResult>();
            using var img = Image.Load<Rgb24>(inputPath);
            originalWidth = img.Width;
            originalHeight = img.Height;

            // 1. Debug: Save what the model sees
            using var debugImg = img.Clone(x => x.Resize(TargetSize, TargetSize));
            debugImg.Save("debug_input.jpg");

            using var session = new InferenceSession(modelPath);
            img.Mutate(x => x.Resize(TargetSize, TargetSize));

            var yoloTensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });
            for (int y = 0; y < TargetSize; y++)
            {
                for (int x = 0; x < TargetSize; x++)
                {
                    var pixel = img[x, y];
                    yoloTensor[0, 0, y, x] = pixel.R / 255f;
                    yoloTensor[0, 1, y, x] = pixel.G / 255f;
                    yoloTensor[0, 2, y, x] = pixel.B / 255f;
                }
            }

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(session.InputMetadata.Keys.First(), yoloTensor) };
            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();

            // 2. Coordinate Scaling Logic
            float scaleX = (float)originalWidth / TargetSize;
            float scaleY = (float)originalHeight / TargetSize;

            for (int i = 0; i < output.Dimensions[1]; i++)
            {
                float conf = output[0, i, 4];
                if (conf < ConfidenceThreshold) continue;

                // YOLO outputs [x1, y1, x2, y2, confidence, class_id]
                // Map 640x640 coordinates back to original image size
                float x1 = output[0, i, 0] * scaleX;
                float y1 = output[0, i, 1] * scaleY;
                float x2 = output[0, i, 2] * scaleX;
                float y2 = output[0, i, 3] * scaleY;

                detections.Add(new DetectionResult
                {
                    X = x1,
                    Y = y1,
                    Width = x2 - x1,
                    Height = y2 - y1,
                    Confidence = conf,
                    Label = "Pothole"
                });
            }
            return detections;
        }


        //public List<DetectionResult> Detect(string modelPath, string inputPath)
        //{
        //    var detections = new List<DetectionResult>();
        //    using var session = new InferenceSession(modelPath);
        //    using var img = Image.Load<Rgb24>(inputPath);

        //    // Resize to 640x640 for YOLO
        //    img.Mutate(x => x.Resize(TargetSize, TargetSize));

        //    var yoloTensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });

        //    // Ensure normalization matches common YOLO standards (0-1)
        //    for (int y = 0; y < TargetSize; y++)
        //    {
        //        for (int x = 0; x < TargetSize; x++)
        //        {
        //            var pixel = img[x, y];
        //            yoloTensor[0, 0, y, x] = pixel.R / 255f;
        //            yoloTensor[0, 1, y, x] = pixel.G / 255f;
        //            yoloTensor[0, 2, y, x] = pixel.B / 255f;
        //        }
        //    }

        //    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(session.InputMetadata.Keys.First(), yoloTensor) };
        //    using var results = session.Run(inputs);
        //    var output = results.First().AsTensor<float>();

        //    // DIAGNOSTIC BLOCK: Print first 3 results to verify parsing
        //    Console.WriteLine("[Detector Debug] First 3 Raw Detections:");
        //    for (int i = 0; i < 3; i++)
        //    {
        //        Console.WriteLine($"Box {i}: x={output[0, i, 0]:F2}, y={output[0, i, 1]:F2}, w={output[0, i, 2]:F2}, h={output[0, i, 3]:F2}, conf={output[0, i, 4]:F4}");
        //    }

        //    //For Pothole Detection: Filter by confidence and class ID (assuming class ID 0 is "pothole")
        //    for (int i = 0; i < output.Dimensions[1]; i++)
        //    {
        //        // YOLO output: [x, y, w, h, confidence, class_id]
        //        float confidence = output[0, i, 4];
        //        int classId = (int)output[0, i, 5];

        //        // Assuming your pothole model uses index 0 for "pothole"
        //        if (confidence < ConfidenceThreshold || classId != 0) continue;

        //        detections.Add(new DetectionResult
        //        {
        //            X = output[0, i, 0],
        //            Y = output[0, i, 1],
        //            Width = output[0, i, 2],
        //            Height = output[0, i, 3],
        //            Confidence = confidence
        //        });
        //    }

        //    //for (int i = 0; i < output.Dimensions[1]; i++)
        //    //{
        //    //    // Check if confidence at index 4 is valid
        //    //    if (output[0, i, 4] < ConfidenceThreshold) continue;

        //    //    detections.Add(new DetectionResult
        //    //    {
        //    //        X = output[0, i, 0],
        //    //        Y = output[0, i, 1],
        //    //        Width = output[0, i, 2],
        //    //        Height = output[0, i, 3],
        //    //        Confidence = output[0, i, 4]
        //    //    });
        //    //}

        //    return detections;
        //} 



        //public List<DetectionResult> Detect(string modelPath, string inputPath)
        //{
        //    var detections = new List<DetectionResult>();
        //    using var session = new InferenceSession(modelPath);
        //    using var img = Image.Load<Rgb24>(inputPath);

        //    // ... (preprocessing omitted for brevity, same as previous logic)
        //    // Assume tensor created...
        //    var yoloTensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });
        //    // ... (tensor filling logic)

        //    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(session.InputMetadata.Keys.First(), yoloTensor) };
        //    using var results = session.Run(inputs);
        //    var output = results.First().AsTensor<float>();

        //    for (int i = 0; i < output.Dimensions[1]; i++)
        //    {
        //        if (output[0, i, 4] < ConfidenceThreshold) continue;

        //        detections.Add(new DetectionResult
        //        {
        //            X = output[0, i, 0],
        //            Y = output[0, i, 1],
        //            Width = output[0, i, 2],
        //            Height = output[0, i, 3],
        //            Confidence = output[0, i, 4]
        //        });
        //    }
        //    return detections;
        //}
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

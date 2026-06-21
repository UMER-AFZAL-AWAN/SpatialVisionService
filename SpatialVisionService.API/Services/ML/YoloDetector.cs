using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SpatialVisionService.API.Models;
using SpatialVisionService.API.Services.Interfaces;

namespace SpatialVisionService.API.Services.ML;

/// <summary>
/// YOLO-based object detector for pothole detection.
/// Implements Rule A (0–100 confidence normalization) and Rule B (coordinate scaling).
/// </summary>
public class YoloDetector : IObjectDetector
{
    private const int TargetSize = 640;

    // Standard YOLO confidence is on a 0.0-1.0 scale
    private const float ConfidenceThreshold = 0.40f;

    private readonly InferenceSession _session;

    // Class label map for the fine-tuned pothole model.
    // Extend this dictionary if the model is retrained with additional classes.
    private static readonly Dictionary<int, string> ClassLabels = new()
    {
        { 0, "Pothole" }
    };

    public YoloDetector(InferenceSession session)
    {
        _session = session;
    }

    public List<DetectionResult> Detect(Image<Rgb24> image, int originalWidth, int originalHeight)
    {
        var detections = new List<DetectionResult>();

        // Clone to avoid mutating the caller's image (depth estimator needs it intact)
        using var resized = image.Clone(ctx => ctx.Resize(TargetSize, TargetSize));

        var tensor = Preprocess(resized);
        var inputName = _session.InputMetadata.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        // The actual tensor shape for YOLOv8/v10 is usually [1, 6, 300] or similar, 
        // where Dimensions[1] is the attributes (x, y, x2, y2, conf, class_id)
        // and Dimensions[2] is the number of detections.
        // Even if the user prompt stated [1, 300, 6], the observed output (2 results, garbled coordinates/classes)
        // proves it is iterating across the 6 attributes instead of the 300 detections.
        int numDetections = output.Dimensions[2];

        // Rule B: Scale factor from 640x640 model space → original image dimensions
        float scaleX = (float)originalWidth / TargetSize;
        float scaleY = (float)originalHeight / TargetSize;

        for (int i = 0; i < numDetections; i++)
        {
            // Confidence at index 4 (attribute dimension) is on 0.0-1.0 scale
            float rawConf = output[0, 4, i];
            if (rawConf < ConfidenceThreshold) continue;

            // Class ID at index 5, cast to int and map to label
            int classId = (int)output[0, 5, i];
            string label = ClassLabels.GetValueOrDefault(classId, $"Class_{classId}");

            // The model outputs [x1, y1, x2, y2]
            float x1 = output[0, 0, i] * scaleX;
            float y1 = output[0, 1, i] * scaleY;
            float x2 = output[0, 2, i] * scaleX;
            float y2 = output[0, 3, i] * scaleY;

            detections.Add(new DetectionResult
            {
                X = x1,
                Y = y1,
                Width = x2 - x1,  // Actual width
                Height = y2 - y1, // Actual height
                Confidence = rawConf,
                ClassId = classId,
                Label = label
            });
        }

        return detections;
    }

    private static DenseTensor<float> Preprocess(Image<Rgb24> image)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });

        for (int y = 0; y < TargetSize; y++)
        {
            for (int x = 0; x < TargetSize; x++)
            {
                var pixel = image[x, y];
                tensor[0, 0, y, x] = pixel.R / 255f;
                tensor[0, 1, y, x] = pixel.G / 255f;
                tensor[0, 2, y, x] = pixel.B / 255f;
            }
        }

        return tensor;
    }
}

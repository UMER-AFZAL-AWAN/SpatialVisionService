using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SpatialVisionService.API.Services.Interfaces;

namespace SpatialVisionService.API.Services.ML;

/// <summary>
/// DPT/MiDaS-based monocular depth estimator.
/// Produces a 384×384 relative depth map from a single RGB image.
/// </summary>
public class DptEstimator : IDepthEstimator
{
    private const int TargetSize = 384;

    // ImageNet normalization constants
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std  = { 0.229f, 0.224f, 0.225f };

    private readonly InferenceSession _session;

    public DptEstimator(InferenceSession session)
    {
        _session = session;
    }

    public float[] GetDepthMap(Image<Rgb24> image)
    {
        // Clone to avoid mutating the caller's image
        using var resized = image.Clone(ctx => ctx.Resize(TargetSize, TargetSize));

        var tensor = Preprocess(resized);
        var inputName = _session.InputMetadata.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };

        using var results = _session.Run(inputs);
        return results.First().AsTensor<float>().ToArray();
    }

    private static DenseTensor<float> Preprocess(Image<Rgb24> image)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });

        for (int y = 0; y < TargetSize; y++)
        {
            for (int x = 0; x < TargetSize; x++)
            {
                var pixel = image[x, y];
                tensor[0, 0, y, x] = ((pixel.R / 255f) - Mean[0]) / Std[0];
                tensor[0, 1, y, x] = ((pixel.G / 255f) - Mean[1]) / Std[1];
                tensor[0, 2, y, x] = ((pixel.B / 255f) - Mean[2]) / Std[2];
            }
        }

        return tensor;
    }
}

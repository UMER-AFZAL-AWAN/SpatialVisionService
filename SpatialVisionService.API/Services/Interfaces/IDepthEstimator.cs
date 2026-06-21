using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SpatialVisionService.API.Services.Interfaces;

/// <summary>
/// Estimates per-pixel depth from a single image, returning a flat depth map array.
/// </summary>
public interface IDepthEstimator
{
    float[] GetDepthMap(Image<Rgb24> image);
}

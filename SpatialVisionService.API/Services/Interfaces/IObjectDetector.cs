using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SpatialVisionService.API.Models;

namespace SpatialVisionService.API.Services.Interfaces;

/// <summary>
/// Detects objects in an image and returns bounding boxes scaled to original dimensions.
/// </summary>
public interface IObjectDetector
{
    List<DetectionResult> Detect(Image<Rgb24> image, int originalWidth, int originalHeight);
}

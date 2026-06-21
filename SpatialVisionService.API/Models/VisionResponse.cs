namespace SpatialVisionService.API.Models;

/// <summary>
/// API response wrapper for the spatial vision analysis endpoint.
/// </summary>
public class VisionResponse
{
    /// <summary>List of detected objects with spatial depth data.</summary>
    public List<DetectionResult> Detections { get; set; } = new();

    /// <summary>Original image width in pixels.</summary>
    public int ImageWidth { get; set; }

    /// <summary>Original image height in pixels.</summary>
    public int ImageHeight { get; set; }

    /// <summary>Total pipeline processing time in milliseconds.</summary>
    public double ProcessingTimeMs { get; set; }
}

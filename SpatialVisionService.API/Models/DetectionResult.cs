namespace SpatialVisionService.API.Models;

/// <summary>
/// Represents a single detected object with its spatial telemetry.
/// </summary>
public class DetectionResult
{
    /// <summary>Top-left X coordinate in original image space.</summary>
    public float X { get; set; }

    /// <summary>Top-left Y coordinate in original image space.</summary>
    public float Y { get; set; }

    /// <summary>Bounding box width in original image space.</summary>
    public float Width { get; set; }

    /// <summary>Bounding box height in original image space.</summary>
    public float Height { get; set; }

    /// <summary>Detection confidence, normalized to 0.0–1.0.</summary>
    public float Confidence { get; set; }

    /// <summary>Raw class ID from the model output.</summary>
    public int ClassId { get; set; }

    /// <summary>Human-readable label mapped from ClassId.</summary>
    public string Label { get; set; } = "Unknown";

    /// <summary>Depth value sampled from the DPT depth map at the detection center.</summary>
    public float DepthValue { get; set; }
}

using Microsoft.AspNetCore.Mvc;
using SpatialVisionService.API.Models;
using SpatialVisionService.API.Services.Interfaces;

namespace SpatialVisionService.API.Controllers;

/// <summary>
/// Endpoint for spatial vision analysis — pothole detection with depth mapping.
/// </summary>
[ApiController]
[Route("api/vision")]
public class VisionController : ControllerBase
{
    private readonly ISpatialOrchestrator _orchestrator;
    private readonly ILogger<VisionController> _logger;

    private static readonly HashSet<string> AllowedContentTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp"
    };

    private const long MaxFileSize = 10 * 1024 * 1024; // 10 MB

    public VisionController(ISpatialOrchestrator orchestrator, ILogger<VisionController> logger)
    {
        _orchestrator = orchestrator;
        _logger = logger;
    }

    /// <summary>
    /// Analyzes an uploaded image for pothole detection with spatial depth mapping.
    /// Accepts a single image file via multipart/form-data.
    /// </summary>
    /// <param name="image">The image file to analyze (JPEG, PNG, BMP, or WebP, max 10 MB).</param>
    /// <returns>A <see cref="VisionResponse"/> containing detected objects with depth values.</returns>
    [HttpPost("analyze")]
    [ProducesResponseType(typeof(VisionResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> Analyze(IFormFile image)
    {
        // --- Input Validation ---
        if (image is null || image.Length == 0)
        {
            return BadRequest(new { error = "No image file provided." });
        }

        if (!AllowedContentTypes.Contains(image.ContentType))
        {
            return BadRequest(new
            {
                error = $"Unsupported content type '{image.ContentType}'.",
                allowed = AllowedContentTypes
            });
        }

        if (image.Length > MaxFileSize)
        {
            return BadRequest(new
            {
                error = $"File size ({image.Length / (1024 * 1024.0):F1} MB) exceeds the {MaxFileSize / (1024 * 1024)} MB limit."
            });
        }

        _logger.LogInformation("Received: {FileName} ({Size} bytes, {Type})",
            image.FileName, image.Length, image.ContentType);

        // --- Delegate to the orchestrator ---
        using var stream = image.OpenReadStream();
        var response = await _orchestrator.AnalyzeAsync(stream);

        return Ok(response);
    }
}

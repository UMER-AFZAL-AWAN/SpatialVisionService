using SpatialVisionService.API.Models;

namespace SpatialVisionService.API.Services.Interfaces;

/// <summary>
/// Orchestrates the full spatial vision pipeline: detection → depth → fusion.
/// </summary>
public interface ISpatialOrchestrator
{
    Task<VisionResponse> AnalyzeAsync(Stream imageStream);
}

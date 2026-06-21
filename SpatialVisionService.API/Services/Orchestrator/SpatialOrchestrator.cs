using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SpatialVisionService.API.Models;
using SpatialVisionService.API.Services.Interfaces;
using System.Diagnostics;
using System.IO;

namespace SpatialVisionService.API.Services.Orchestrator;

/// <summary>
/// Coordinates the ML services and performs the Spatial Fusion math.
/// Implements Rule C: maps YOLO detections to the DPT depth grid using original image dimensions.
/// </summary>
public class SpatialOrchestrator : ISpatialOrchestrator
{
    private const int DepthGridSize = 384;

    private readonly IObjectDetector _detector;
    private readonly IDepthEstimator _depthEstimator;
    private readonly ILogger<SpatialOrchestrator> _logger;

    public SpatialOrchestrator(
        IObjectDetector detector,
        IDepthEstimator depthEstimator,
        ILogger<SpatialOrchestrator> logger)
    {
        _detector = detector;
        _depthEstimator = depthEstimator;
        _logger = logger;
    }

    public async Task<VisionResponse> AnalyzeAsync(Stream imageStream)
    {
        var sw = Stopwatch.StartNew();

        // Load the image once; both pipelines receive the same immutable reference
        using var image = await Image.LoadAsync<Rgb24>(imageStream);
        int originalWidth = image.Width;
        int originalHeight = image.Height;

        _logger.LogInformation("Processing image: {Width}x{Height}", originalWidth, originalHeight);

        // --- Pipeline Stage 1: Depth Estimation ---
        _logger.LogInformation("Running depth estimation (DPT 384x384)...");
        float[] depthMap = _depthEstimator.GetDepthMap(image);

        // --- Pipeline Stage 2: Object Detection ---
        _logger.LogInformation("Running object detection (YOLO 640x640)...");
        var detections = _detector.Detect(image, originalWidth, originalHeight);

        _logger.LogInformation("Detected {Count} object(s). Performing spatial fusion...", detections.Count);

        Font font = SystemFonts.TryGet("Arial", out FontFamily family) 
            ? new Font(family, 16) 
            : SystemFonts.Families.First().CreateFont(16);

        // --- Pipeline Stage 3: Spatial Fusion (Rule C) ---
        foreach (var detection in detections)
        {
            // 1. Get center in original image space
            float centerX = detection.X + (detection.Width / 2f);
            float centerY = detection.Y + (detection.Height / 2f);

            // 2. Map to 384x384 depth grid using ORIGINAL dimensions (Rule C — CRITICAL FIX)
            //    Dividing by originalWidth/Height gives the percentage offset, which
            //    is then applied to the 384 grid. This avoids the aspect-ratio drift bug
            //    that occurred when dividing by 640.
            int gridX = Math.Clamp((int)((centerX / originalWidth) * DepthGridSize), 0, DepthGridSize - 1);
            int gridY = Math.Clamp((int)((centerY / originalHeight) * DepthGridSize), 0, DepthGridSize - 1);

            // 3. Extract Z-Depth from the flat depth map array
            detection.DepthValue = depthMap[gridY * DepthGridSize + gridX];

            // 4. Draw overlay onto the image
            var rect = new RectangleF(detection.X, detection.Y, detection.Width, detection.Height);
            image.Mutate(ctx => ctx.Draw(Color.LimeGreen, 2f, rect));
            
            string label = $"{detection.Label} | D:{detection.DepthValue:F2}";
            image.Mutate(ctx => ctx.DrawText(label, font, Color.White, new PointF(detection.X, Math.Max(0, detection.Y - 20))));

            _logger.LogDebug(
                "  → '{Label}' center=({CX:F0},{CY:F0}) → grid=({GX},{GY}) → depth={Depth:F4}",
                detection.Label, centerX, centerY, gridX, gridY, detection.DepthValue);
        }

        // --- Pipeline Stage 4: Encode Image ---
        using var ms = new MemoryStream();
        await image.SaveAsJpegAsync(ms);
        string base64Image = Convert.ToBase64String(ms.ToArray());

        sw.Stop();
        _logger.LogInformation("Pipeline complete in {Elapsed:F1}ms", sw.Elapsed.TotalMilliseconds);

        return new VisionResponse
        {
            Detections = detections,
            ImageWidth = originalWidth,
            ImageHeight = originalHeight,
            ProcessingTimeMs = sw.Elapsed.TotalMilliseconds,
            AnnotatedImageBase64 = base64Image
        };
    }
}

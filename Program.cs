using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SpatialVisionService.Orchestrator;
using SpatialVisionService.Services;

namespace SpatialVisionService;

public class DetectionResult
{
    public string Label { get; set; } = "Unknown";
    public float Confidence { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float Width { get; set; }
    public float Height { get; set; }
    public float DepthValue { get; set; } // We will populate this in the orchestrator
}
internal class Program
{
    static void Main(string[] args)
    {
        string root = AppContext.BaseDirectory;
        var orchestrator = new SpatialOrchestrator();

        orchestrator.RunPipeline(
            Path.Combine(root, "Models", "dpt_large_384.onnx"),
            Path.Combine(root, "Models", "FineTunedPotHole.onnx"),
            Path.Combine(root, "input.jpg")
        );
    }
    //static void Main(string[] args)
    //{
    //    // Establish environment paths relative to execution bins
    //    string baseDir = AppContext.BaseDirectory; 
    //    string projectRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", ".."));

    //    string dptModelPath = Path.Combine(projectRoot, "Models", "dpt_large_384.onnx");
    //    string yoloModelPath = Path.Combine(projectRoot, "Models", "yolo26l.onnx");

    //    string inputImagePath = Path.Combine(projectRoot, "input.jpg");
    //    string depthOutputPath = Path.Combine(projectRoot, "depth_output.png");
    //    string yoloOutputPath = Path.Combine(projectRoot, "yolo_output.jpg");

    //    // Infrastructure check
    //    if (!File.Exists(dptModelPath))
    //    {
    //        Console.WriteLine($"Error: Core depth model missing at target path: {dptModelPath}");
    //        return;
    //    }

    //    if (!File.Exists(inputImagePath))
    //    {
    //        CreateDummyImage(inputImagePath);
    //    }

    //    try
    //    {
    //        // Execute decoupled pipelines
    //        var depthPipeline = new DepthEstimator();
    //        depthPipeline.Process(dptModelPath, inputImagePath, depthOutputPath);

    //        var detectionPipeline = new ObjectDetector();
    //        detectionPipeline.Process(yoloModelPath, inputImagePath, yoloOutputPath);
    //    }
    //    catch (Exception ex)
    //    {
    //        Console.WriteLine($"\nPipeline execution halted with structural failure: {ex.Message}");
    //    }
    //}

    private static void CreateDummyImage(string path)
    {
        using var img = new Image<Rgb24>(640, 640);
        img.Mutate(ctx => ctx.ProcessPixelRowsAsVector4((row) => {
            for (int x = 0; x < row.Length; x++)
            {
                row[x].X = x / 640f;
                row[x].Y = 0.5f;
                row[x].Z = 1f - (x / 640f);
            }
        }));
        img.SaveAsJpeg(path);
        Console.WriteLine($"Generated initial baseline canvas frame at: {path}");
    }
}
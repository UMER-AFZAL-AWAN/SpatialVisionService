using Microsoft.ML.OnnxRuntime;
using SpatialVisionService.API.Services.Interfaces;
using SpatialVisionService.API.Services.ML;
using SpatialVisionService.API.Services.Orchestrator;

var builder = WebApplication.CreateBuilder(args);

// --- Service Registration ---
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// --- ONNX Model Configuration ---
// Resolve model paths from appsettings.json, relative to the content root
var yoloModelPath = builder.Configuration["ModelPaths:YoloModel"]
    ?? throw new InvalidOperationException("ModelPaths:YoloModel is not configured in appsettings.json.");
var dptModelPath = builder.Configuration["ModelPaths:DptModel"]
    ?? throw new InvalidOperationException("ModelPaths:DptModel is not configured in appsettings.json.");

yoloModelPath = Path.GetFullPath(yoloModelPath, builder.Environment.ContentRootPath);
dptModelPath = Path.GetFullPath(dptModelPath, builder.Environment.ContentRootPath);

// Register InferenceSession instances as Singletons (keyed services, .NET 8).
// Models are loaded ONCE at startup and shared across all requests.
// InferenceSession.Run() is thread-safe.
builder.Services.AddKeyedSingleton<InferenceSession>("yolo", (_, _) =>
{
    Console.WriteLine($"[Startup] Loading YOLO model from: {yoloModelPath}");
    return new InferenceSession(yoloModelPath);
});

builder.Services.AddKeyedSingleton<InferenceSession>("dpt", (_, _) =>
{
    Console.WriteLine($"[Startup] Loading DPT model from: {dptModelPath}");
    return new InferenceSession(dptModelPath);
});

// --- ML Service Registration ---
builder.Services.AddSingleton<IObjectDetector>(sp =>
    new YoloDetector(sp.GetRequiredKeyedService<InferenceSession>("yolo")));

builder.Services.AddSingleton<IDepthEstimator>(sp =>
    new DptEstimator(sp.GetRequiredKeyedService<InferenceSession>("dpt")));

builder.Services.AddScoped<ISpatialOrchestrator, SpatialOrchestrator>();

// --- Build & Configure Pipeline ---
var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapControllers();

Console.WriteLine("===========================================");
Console.WriteLine("  Spatial Vision API — Ready");
Console.WriteLine("  POST /api/vision/analyze");
Console.WriteLine("  Swagger: http://localhost:5000/swagger");
Console.WriteLine("===========================================");

app.Run();

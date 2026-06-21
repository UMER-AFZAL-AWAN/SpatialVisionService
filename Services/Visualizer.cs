using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SpatialVisionService.Services
{
    public static class Visualizer
    {
        public static void RenderResults(string inputPath, string outputPath, List<DetectionResult> detections)
        {
            using var image = Image.Load<Rgb24>(inputPath);

            // Load a standard font (Ensure you have a font file in your project or use System fonts)
            // If this throws an error, you can use a default font: SystemFonts.CreateFont("Arial", 16)
            var font = SystemFonts.CreateFont("Arial", 16);

            foreach (var obj in detections)
            {
                // Convert normalized coordinates back to image scale
                var rect = new RectangleF(obj.X, obj.Y, obj.Width, obj.Height);

                // Draw Box
                image.Mutate(ctx => ctx.Draw(Color.LimeGreen, 2f, rect));

                // Draw Text
                string label = $"{obj.Label} | D:{obj.DepthValue:F2}";
                image.Mutate(ctx => ctx.DrawText(label, font, Color.White, new PointF(obj.X, obj.Y - 20)));
            }

            image.SaveAsJpeg(outputPath);
            Console.WriteLine($"[Visualizer] Saved overlay image to: {outputPath}");
        }
    }
}

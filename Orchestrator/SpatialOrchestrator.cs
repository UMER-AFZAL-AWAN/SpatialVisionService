using SpatialVisionService.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SpatialVisionService.Orchestrator
{
    public class SpatialOrchestrator
    {
        private readonly DepthEstimator _depth = new();
        private readonly ObjectDetector _detector = new();

        public void RunPipeline(string dptModel, string yoloModel, string imagePath)
        {
            Console.WriteLine("[Orchestrator] Starting Pipeline...");

            // 1. Run inference
            Console.WriteLine("[Orchestrator] Running Depth Estimation...");
            var depthMap = _depth.GetDepthMap(dptModel, imagePath);

            Console.WriteLine("[Orchestrator] Running Object Detection...");
            //var objects = _detector.Detect(yoloModel, imagePath);
            var potholes = _detector.Detect(yoloModel, imagePath, out int width, out int height);

            // 2. Spatial Mapping
            if (potholes.Count == 0)
            {
                Console.WriteLine("[Orchestrator] WARNING: No objects detected. Check if input image is valid or threshold is too high.");
                return;
            }

            Console.WriteLine($"[Orchestrator] Mapping {potholes.Count} objects to depth map...");
            foreach (var hole in potholes)
            {
                // 1. Get the center of the pothole (Already in Original Image Coordinates)
                int centerX = (int)(hole.X + (hole.Width / 2));
                int centerY = (int)(hole.Y + (hole.Height / 2));

                // 2. Map to 384x384 depth grid using Original Dimensions
                // Use (float)width and (float)height instead of 640f
                int gridX = Math.Clamp((int)((centerX / (float)width) * 384), 0, 383);
                int gridY = Math.Clamp((int)((centerY / (float)height) * 384), 0, 383);

                // 3. Get Depth
                float holeDepthValue = depthMap[gridY * 384 + gridX];

                Console.WriteLine($"Pothole Detected! Position: ({centerX}, {centerY})");
                Console.WriteLine($"Mapped to Grid: ({gridX}, {gridY}) | Depth: {holeDepthValue:F4}");

                // ... rest of your logic
            }
            //foreach (var hole in potholes)
            //{
            //    // 1. Get the center of the pothole
            //    int centerX = (int)(hole.X + (hole.Width / 2));
            //    int centerY = (int)(hole.Y + (hole.Height / 2));

            //    // 2. Map to 384x384 depth grid
            //    int gridX = Math.Clamp((int)((centerX / 640f) * 384), 0, 383);
            //    int gridY = Math.Clamp((int)((centerY / 640f) * 384), 0, 383);

            //    // 3. Get Depth
            //    float holeDepthValue = depthMap[gridY * 384 + gridX];

            //    // 4. Heuristic for "Pothole Severity"
            //    // Since a pothole is further away than the surrounding road, 
            //    // the depth value will be slightly higher than the surrounding road.
            //    Console.WriteLine($"Pothole Detected! Position: ({centerX}, {centerY})");
            //    Console.WriteLine($"Relative Distance Value: {holeDepthValue:F4}");

            //    // This is a placeholder for your alert logic
            //    if (holeDepthValue > 20.0f)
            //    {
            //        Console.WriteLine("ALERT: High severity pothole detected.");
            //    }
            //}

            //foreach (var obj in objects)
            //{
            //    // Map object center to 384x384 depth grid
            //    int gridX = (int)((obj.X / 640f) * 384);
            //    int gridY = (int)((obj.Y / 640f) * 384);

            //    // Safety clamp
            //    gridX = Math.Clamp(gridX, 0, 383);
            //    gridY = Math.Clamp(gridY, 0, 383);

            //    obj.DepthValue = depthMap[gridY * 384 + gridX];

            //    Console.WriteLine($"-> Detected Object at X:{obj.X:F0}, Y:{obj.Y:F0} | Confidence: {obj.Confidence:P0} | Depth Score: {obj.DepthValue:F4}");
            //}

            // Save the visualization
            Visualizer.RenderResults(imagePath, "output_annotated.jpg", potholes);

            Console.WriteLine("[Orchestrator] Pipeline Complete. Check output_annotated.jpg for results.");
        }
    }
}

using Emgu.CV;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using System;
using System.ComponentModel;

string? inputPath = null;
string? outputPath = null;

for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "-i" && i + 1 < args.Length)
    {
        inputPath = args[++i];
    }
    else if (args[i] == "-o" && i + 1 < args.Length)
    {
        outputPath = args[++i];
    }
}
if (inputPath == null || outputPath == null)
{
    Console.WriteLine("Usage: program -i <input video> -o <output file>");
    return;
}

List<byte> result = [];

try
{
    VideoCapture capture = new(inputPath);
    if (!capture.IsOpened)
    {
        Console.WriteLine("Error opening video file.");
        return;
    }

    int width = (int)capture.Get(Emgu.CV.CvEnum.CapProp.FrameWidth);
    int height = (int)capture.Get(Emgu.CV.CvEnum.CapProp.FrameHeight);
    float fps = (float)capture.Get(Emgu.CV.CvEnum.CapProp.Fps);

    result.AddRange(BitConverter.GetBytes(width));
    result.AddRange(BitConverter.GetBytes(height));
    result.AddRange(BitConverter.GetBytes(fps));

    Mat frame = new();
    Mat grayFrame = new();
    Mat thresholdedFrame = new();

    while (capture.Read(frame))
    {
        if (frame.IsEmpty) break;

        CvInvoke.CvtColor(frame, grayFrame, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
        CvInvoke.Threshold(grayFrame, thresholdedFrame, 128, 255, Emgu.CV.CvEnum.ThresholdType.Binary);

        Image<Gray, byte> grayImage = thresholdedFrame.ToImage<Gray, byte>();

        int totalPixels = width * height;
        byte packedByte = 0;
        int bitCount = 0;

        for (int i = 0; i < totalPixels; i++)
        {
            int y = i / width;
            int x = i % width;

            byte pixelValue = grayImage.Data[y, x, 0];
            bool isWhite = pixelValue == 255;

            packedByte |= (byte)((isWhite ? 1 : 0) << (7 - bitCount));

            if (++bitCount == 8)
            {
                result.Add(packedByte);
                packedByte = 0;
                bitCount = 0;
            }
        }

        if (bitCount > 0) result.Add(packedByte);

    }   

    capture.Dispose();

    File.WriteAllBytes(outputPath, [.. result]);
    Console.WriteLine($"Preprocessed binary saved to {outputPath}");
}
catch (Exception ex)
{
    Console.WriteLine("An error occurred: " + ex.Message);
}
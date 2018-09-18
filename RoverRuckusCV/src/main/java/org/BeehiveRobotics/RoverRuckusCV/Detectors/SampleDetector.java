package org.BeehiveRobotics.RoverRuckusCV.Detectors;

import org.BeehiveRobotics.RoverRuckusCV.OpenCVPipeline;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Scalar;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;

import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

//This will detect which position the gold ball is in
public class SampleDetector extends OpenCVPipeline {
    enum CubePosition {
        LEFT, CENTER, RIGHT
    }
    private boolean showContours = true;
    private List<MatOfPoint> contours = new ArrayList<>();
    private Mat hsv = new Mat();
    private Mat threshold = new Mat();

    public synchronized void showContours(boolean enabled) {
        showContours = enabled;
    }
    @Override
    public Mat processFrame(Mat rgba, Mat grayscale) {
        //Convert RGB to HSV, because it makes sense for yellow vs. white
        Imgproc.cvtColor(rgba, hsv, Imgproc.COLOR_RGB2HSV, 3);
        
        //Find everything in a range of HSV values. H is [0, 179], S is [0,255], V is [0,255].
        Core.inRange(hsv, new Scalar(20, 127, 80), new Scalar(40, 255, 120), threshold);
        
        // Blur the image to remove noise, and put the overall shape together
        Imgproc.blur(threshold, threshold, new Size(3, 3));

        // Fills contour list with outlines of yellow objects
        Imgproc.findContours(threshold, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        if(showContours) {
            // Draws the outlines of the yellow over the image on the screen. Colored blue
            Imgproc.drawContours(rgba, contours, -1, new Scalar(0, 0, 255), 2, 8);
        }
        return rgba;

    }
}
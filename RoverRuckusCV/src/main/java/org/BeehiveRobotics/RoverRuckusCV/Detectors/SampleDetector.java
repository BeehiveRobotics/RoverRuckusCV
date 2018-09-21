package org.BeehiveRobotics.RoverRuckusCV.Detectors;

import android.content.Context;

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

    // CALIBRATE THESE NUMBERS FOR YOUR TEAM
    private final double MIN_SIZE = 0;

    
    enum CubePosition {
        LEFT, CENTER, RIGHT, UNKNOWN;
    }
    private boolean showContours = true;
    private boolean showRectangles = true;
    private Mat hsv = new Mat();
    private ArrayList<MatOfPoint> yellowContours = new ArrayList<>();
    private ArrayList<MatOfPoint> whiteContours = new ArrayList<>();
    private ArrayList<MatOfPoint> yellowContoursFiltered = new ArrayList<>();
    private ArrayList<MatOfPoint> whiteContoursFiltered = new ArrayList<>();
    private ArrayList<Double> yellowYValues = new ArrayList<>();
    private ArrayList<Double> whiteYValues = new ArrayList<>();
    private Mat yellowThreshold = new Mat();
    private Mat whiteThreshold = new Mat();
    private ArrayList<Double> sizes = new ArrayList<>();
    private CubePosition currentCubePosition = UNKNOWN;
    
    public synchronized void showContours(boolean enabled) {
        showContours = enabled;
    }
    public synchronized void showRectangles(boolean enabled) {
        showRectangles = enabled;
    }
    @Override
    public Mat processFrame(Mat rgba, Mat grayscale) {
        Size matSize = rgba.size();
        
        //Convert RGB to HSV, because it makes sense for yellow vs. white
        Imgproc.cvtColor(rgba, hsv, Imgproc.COLOR_RGB2HSV, 3);
        
        //Find everything in a range of HSV values. H is [0, 179], S is [0,255], V is [0,255].
        //This numbers may need to be changed depending on lighting, camera, etc.
        //First is yellow, second is white
        //The numbers were calibrated from a Nexus 5 phone, in the lighting at our school.
        Core.inRange(hsv, new Scalar(20, 80, 60), new Scalar(32, 255, 255), yellowThreshold);
        Core.inRange(hsv, new Scalar(0, 0, 140), new Scalar(179, 15, 255), whiteThreshold);
        
        // Blur the image to remove noise, and put the overall shape together
        Imgproc.blur(yellowThreshold, yellowThreshold, new Size(3, 3));
        Imgproc.blur(whiteThreshold, whiteThreshold, new Size(3, 3));

        // Clear all contour lists
        yellowContours.clear();
        whiteContours.clear();
        yellowContoursFiltered.clear();
        whiteContoursFiltered.clear();

        // Fills contour list with outlines of yellow and white
        Imgproc.findContours(yellowThreshold, yellowContours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.findContours(whiteThreshold, whiteContours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        
        yellowContoursFiltered = yellowContours.clone();
        whiteContoursFiltered = whiteContours.clone();

        for(MatOfPoint contour: yellowContoursFiltered) {
            if(contour.size().height > MIN_SIZE) {
                yellowContoursFiltered.remove(contour);
            }
        }
        for(MatOfPoint contour: whiteContoursFiltered) {
            if(contour.size().height > MIN_SIZE) {
                whiteContoursFiltered.remove(contour);
            }
        }
        
        if(showContours) {
            // Draws the outlines of the yellow over the image on the screen. Colored blue.
            Imgproc.drawContours(rgba, yellowContoursFiltered, -1, new Scalar(0, 0, 255), 2, 8);
            // Draws the outlines of the white over the image on the screen. Colored green.
            Imgproc.drawContours(rgba, whiteContoursFiltered, -1, new Scalar(0, 255, 0), 2, 8);
        }
        /*
        for(MatOfPoint contour: yellowContoursFiltered) {
            sizes.add(contour.size().height);
        }
        */

        for(MatOfPoint contour: yellowContoursFiltered) {
            Rect rectangle = Imgproc.boundingRect(c);
            double x = rectangle.x;
            double y = rectangle.y;
            double w = rectangle.width;
            double h = rectangle.height;

            Point centerPoint = new Point(x + ( w/2), y + (h/2));

            if(showRectangles) {
                Imgproc.rectangle(rgba, new Point(x, y), new Point((x+w), (y+h)), new Scalar(0, 0, 255), 1);
            }
        }

        for(MatOfPoint contour: whiteContoursFiltered) {
            Rect rectangle = Imgproc.boundingRect(c);
            double x = rectangle.x;
            double y = rectangle.y;
            double w = rectangle.width;
            double h = rectangle.height;

            Point centerPoint = new Point(x + ( w/2), y + (h/2));

            if(showRectangles) {
                Imgproc.rectangle(rgba, new Point(x, y), new Point((x+w), (y+h)), new Scalar(0, 255, 0), 1);
            }
        }        
        
        return rgba;
    }
    public ArrayList<Double> getSizes() {
        return this.sizes;
    }
    public CubePosition getCubePosition(int framesToTest) {
        int left = 0;
        int middle = 0;
        int right = 0;
        for(int frame = 0; frame < framesToTest; frame++) {
            switch(currentCubePosition) {
                case LEFT: left++;
                case MIDDLE: middle++;
                case RIGHT: right++;
            }
        }
        if(left > middle) {
            if(left > right) {
                return LEFT;
            } else {
                return RIGHT;
            }
        } else {
            if(middle > right) {
                return MIDDLE;
            } else {
                return RIGHT;
            }
        }
    }
    public CubePosition getCubePosition() {return getCubePosition(30);}
}
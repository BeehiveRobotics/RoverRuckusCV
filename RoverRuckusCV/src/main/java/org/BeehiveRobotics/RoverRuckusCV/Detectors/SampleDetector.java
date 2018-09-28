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
import org.opencv.core.Point;
import org.opencv.core.Rect;

import org.opencv.imgproc.Imgproc;

//This will detect which position the gold ball is in
public class SampleDetector extends OpenCVPipeline {

    // CALIBRATE THESE NUMBERS FOR YOUR TEAM
    private final double MIN_BALL_SIZE = 6000;
    private final double MAX_BALL_SIZE = 12000;
    private final double MIN_CUBE_SIZE = 5000;
    private final double MAX_CUBE_SIZE = 10000;

    //These are hsv values. In that order. H, S, V. 
    //H is [0,179], both S and V are [0,255].
    private final Scalar lowYellow  = new Scalar(16,  80,  60);
    private final Scalar highYellow = new Scalar(32,  255, 255);
    private final Scalar lowWhite   = new Scalar(0,   0,   225);
    private final Scalar highWhite  = new Scalar(179, 100, 255);

    //These probably make it slower if they are enabled, but it's helpful for seeing what's going on 
    private boolean showContours   = true;
    private boolean showRectangles = true;

    //In our team, separation anxiety
    private double THRESHOLD_Y_VALUE_FROM_CUBE = 30;
    private double THRESHOLD_SIDE_LENGTH_DIFFERENCE_ALLOWED = 0.2;
    //END OF USER CALIBRATION


    //These shouldn't need changing
    public enum CubePosition {
        LEFT, MIDDLE, RIGHT, UNKNOWN;
    }

    private Mat hsv = new Mat();

    private Size blurSize = new Size(3, 3);

    private final Scalar RED   = new Scalar(255, 0,   0);
    private final Scalar GREEN = new Scalar(0,   255, 0);
    private final Scalar CYAN  = new Scalar(0,   255, 255);
    private final Scalar BLUE  = new Scalar(0,   0,   255);

    private ArrayList<MatOfPoint> yellowContours         = new ArrayList<MatOfPoint>();
    private ArrayList<MatOfPoint> whiteContours          = new ArrayList<MatOfPoint>();
    private ArrayList<MatOfPoint> yellowContoursFiltered = new ArrayList<MatOfPoint>();
    private ArrayList<MatOfPoint> whiteContoursFiltered  = new ArrayList<MatOfPoint>();

    private ArrayList<Rect> yellowRects = new ArrayList<Rect>();
    private ArrayList<Rect> whiteRects  = new ArrayList<Rect>();

    private ArrayList<Double> yellowYValues = new ArrayList<Double>();
    private ArrayList<Double> whiteYValues  = new ArrayList<Double>();

    private Mat yellowThreshold = new Mat();
    private Mat whiteThreshold  = new Mat();

    private CubePosition currentCubePosition = CubePosition.UNKNOWN;

    private int currentFrame = 0;
    
    public synchronized void showContours(boolean enabled) {
        showContours = enabled;
    }

    public synchronized void showRectangles(boolean enabled) {
        showRectangles = enabled;
    }

    @Override
    public Mat processFrame(Mat rgba, Mat grayscale) {
        Size matSize = rgba.size();
        
        //Convert RGB to HSV
        Imgproc.cvtColor(rgba, hsv, Imgproc.COLOR_RGB2HSV, 3);
        
        //Find everything in a range of HSV values. H is [0, 179], S is [0,255], V is [0,255].
        //This numbers may need to be changed depending on lighting, camera, etc.
        //First is yellow, second is white
        //The numbers were calibrated from a Nexus 5 phone, in the lighting at our school.
        Core.inRange(hsv, lowYellow, highYellow, yellowThreshold);
        Core.inRange(hsv, lowWhite, highWhite, whiteThreshold);
        
        // Blur the image to remove noise, and put the overall shape together
        //Imgproc.blur(yellowThreshold, yellowThreshold, blurSize);
        //Imgproc.blur(whiteThreshold, whiteThreshold, blurSize);

        // Clear all lists
        yellowContours.clear();
        whiteContours.clear();
        yellowContoursFiltered.clear();
        whiteContoursFiltered.clear();
        yellowRects.clear();
        whiteRects.clear();

        // Fills contour list with outlines of yellow and white
        Imgproc.findContours(yellowThreshold, yellowContours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.findContours(whiteThreshold, whiteContours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        
        yellowContoursFiltered = new ArrayList<>();
        whiteContoursFiltered = new ArrayList<>();

        //For some reason if you use a foreach (AKA enhanced for loop) here it throws a ConcurrentModificationException
        for(int i = 0; i < yellowContours.size(); i++) {
            MatOfPoint contour = yellowContours.get(i);
            double area = Math.abs(Imgproc.contourArea(contour));
            if(area > MIN_CUBE_SIZE && area < MAX_CUBE_SIZE) {
                yellowContoursFiltered.add(contour);
            }
        }
        for(int i = 0; i < whiteContours.size(); i++) {
            MatOfPoint contour = whiteContours.get(i);
            double area = Math.abs(Imgproc.contourArea(contour));
            if(area > MIN_BALL_SIZE && area < MAX_BALL_SIZE) {
                whiteContoursFiltered.add(contour);
            }
        }
        
        for(MatOfPoint contour: yellowContoursFiltered) {
            Rect rectangle = Imgproc.boundingRect(contour);
            double x = rectangle.x;
            double y = rectangle.y;
            double w = rectangle.width;
            double h = rectangle.height;
            
            double sideRatio = w/h;

            Point centerPoint = new Point(x + (w/2), y + (h/2));

            if(Math.abs(sideRatio - 1) < THRESHOLD_SIDE_LENGTH_DIFFERENCE_ALLOWED) {
                yellowRects.add(rectangle);
            }

            if(showRectangles) {
                Imgproc.rectangle(rgba, new Point(x, y), new Point((x+w), (y+h)), BLUE, 1);
            }
        }

        for(MatOfPoint contour: whiteContoursFiltered) {
            Rect rectangle = Imgproc.boundingRect(contour);
            double x = rectangle.x;
            double y = rectangle.y;
            double w = rectangle.width;
            double h = rectangle.height;
            
            double sideRatio = w/h;
            Point centerPoint = new Point(x + (w/2), y + (h/2));

            if(Math.abs(sideRatio - 1) < THRESHOLD_SIDE_LENGTH_DIFFERENCE_ALLOWED) {
                whiteRects.add(rectangle);
            }

            if(showRectangles) {
                Imgproc.rectangle(rgba, new Point(x, y), new Point((x+w), (y+h)), RED, 1);
            }
        }   
        
        //Start filtering contours based on others positions
        if(yellowRects.size()==1 && whiteRects.size()!=0) {
            final Rect yellowRect = yellowRects.get(0);
            for(int i = 0; i < whiteRects.size(); i++) {
                final Rect whiteRect = whiteRects.get(i);
                if(Math.abs(whiteRect.y - yellowRect.y) > THRESHOLD_Y_VALUE_FROM_CUBE) {
                    whiteRects.remove(i);
                }
            }

        }
        if(whiteRects.size() == 2 && yellowRects.size() != 0) {
            final double yAverage = (whiteRects.get(0).y + whiteRects.get(1).y) / 2;
            for(int i = 0; i < yellowRects.size(); i++) {
                final Rect yellowRect = yellowRects.get(i);
                if(Math.abs(yAverage - yellowRect.y) > THRESHOLD_Y_VALUE_FROM_CUBE) {
                    yellowRects.remove(i);
                }
                
            }
        }
        
        if(yellowRects.size() == 1 && whiteRects.size() == 2) {
            if(yellowRects.get(0).x < whiteRects.get(0).x) {
                if(yellowRects.get(0).x < whiteRects.get(1).x) {
                    currentCubePosition = CubePosition.LEFT;
                } else {
                    currentCubePosition = CubePosition.MIDDLE;
                }
            } else {
                if(yellowRects.get(0).x > whiteRects.get(1).x) {
                    currentCubePosition = CubePosition.RIGHT;
                } else {
                    currentCubePosition = CubePosition.MIDDLE;
                }
            }
            Imgproc.putText(rgba, currentCubePosition.toString(), new Point(30, 30), 0, 1, CYAN);
        }
        if(showContours) {
            // Draws the outlines of the yellow over the image on the screen. Colored blue.
            Imgproc.drawContours(rgba, yellowContoursFiltered, -1, BLUE, 2, 8);
            // Draws the outlines of the white over the image on the screen. Colored red.
            Imgproc.drawContours(rgba, whiteContoursFiltered, -1, RED, 2, 8);
        }
        
        currentFrame++;
        if(currentFrame > 200) {currentFrame = 0;}

        return rgba;
    }

    public CubePosition getCubePosition(int framesToTest) throws InterruptedException {
        int left = 0;
        int middle = 0;
        int right = 0;
        currentFrame = 0;
        for(int frame = 0; frame < framesToTest; frame++) {
            switch(currentCubePosition) {
                case LEFT: left++;
                case MIDDLE: middle++;
                case RIGHT: right++;
                case UNKNOWN: 
            }
            while(currentFrame == frame) {}
        }
        if(left > middle) {
            if(left > right) {
                return CubePosition.LEFT;
            } else {
                return CubePosition.RIGHT;
            }
        } else {
            if(middle > right) {
                return CubePosition.MIDDLE;
            } else {
                return CubePosition.RIGHT;
            }
        }
    }
    public CubePosition getCubePosition() throws InterruptedException {
        return getCubePosition(30);
    }
}
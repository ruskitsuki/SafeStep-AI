#include "PathDetector.h"
#include <vector>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

string PathDetector::detect(Mat frame)
{
    // ðŸš€ 1. à¸¢à¹ˆà¸­à¸‚à¸™à¸²à¸”à¸ à¸²à¸žà¹ƒà¸«à¹‰à¹€à¸«à¸¥à¸·à¸­ 640x480
    Mat workingFrame;
    resize(frame, workingFrame, Size(640, 480));

    Mat hsv, mask;
    Mat debugFrame = workingFrame.clone();

    // --- 1. à¸¥à¸” Noise à¸”à¹‰à¸§à¸¢ Gaussian Blur ---
    Mat blurredFrame;
    GaussianBlur(workingFrame, blurredFrame, Size(3, 3), 0);

    // --- 2. à¹à¸›à¸¥à¸‡à¸ à¸²à¸žà¹€à¸›à¹‡à¸™ HSV ---
    cvtColor(blurredFrame, hsv, COLOR_BGR2HSV);

    // --- 3. à¸„à¸±à¸”à¸à¸£à¸­à¸‡à¸ªà¸µà¹€à¸žà¸·à¹ˆà¸­à¸ªà¸£à¹‰à¸²à¸‡ Mask ---
    Scalar lower(10, 100, 80);
    Scalar upper(35, 255, 255);
    inRange(hsv, lower, upper, mask);

    // --- 4. à¸¥à¸” Noise à¸šà¸™ Mask ---
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    // --- 5. à¸„à¸±à¸”à¹à¸¢à¸à¹€à¸‰à¸žà¸²à¸°à¸à¹‰à¸­à¸™à¸ªà¸µà¸‚à¸²à¸§à¸—à¸µà¹ˆà¹ƒà¸«à¸à¹ˆà¸žà¸­ ---
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat smoothedMask = Mat::zeros(mask.size(), CV_8UC1);

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > 300) {
            drawContours(smoothedMask, contours, i, Scalar(255), FILLED);
        }
    }

    // === à¸¥à¸šà¹€à¸«à¸¥à¸µà¹ˆà¸¢à¸¡à¸‚à¸±à¹‰à¸™à¸šà¸±à¸™à¹„à¸”à¹ƒà¸«à¹‰à¹€à¸›à¹‡à¸™à¸à¹‰à¸­à¸™à¹€à¸™à¸·à¹‰à¸­à¹€à¸”à¸µà¸¢à¸§à¸à¸±à¸™ ===
    GaussianBlur(smoothedMask, smoothedMask, Size(9, 9), 0);
    threshold(smoothedMask, smoothedMask, 127, 255, THRESH_BINARY);
    // ==========================================

    // ðŸš€ === à¸£à¸°à¸šà¸šà¸—à¸µà¹ˆ 1: SLIDING WINDOW (à¸§à¸²à¸”à¸ˆà¸¸à¸”à¸à¸¶à¹ˆà¸‡à¸à¸¥à¸²à¸‡à¸™à¸³à¸—à¸²à¸‡) === ðŸš€
    int nWindows = 10;
    int windowHeight = smoothedMask.rows / nWindows;
    vector<Point> centerPoints;

    for (int i = 0; i < nWindows; i++) {
        int winY = smoothedMask.rows - (i + 1) * windowHeight;
        Rect windowRect(0, winY, smoothedMask.cols, windowHeight);

        Mat windowMask = smoothedMask(windowRect);
        vector<vector<Point>> winContours;
        findContours(windowMask, winContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (size_t j = 0; j < winContours.size(); j++) {
            if (contourArea(winContours[j]) > 30) {
                Moments m = moments(winContours[j]);
                if (m.m00 > 0) {
                    int cx = m.m10 / m.m00;
                    int cy = m.m01 / m.m00 + winY;
                    centerPoints.push_back(Point(cx, cy));
                    // à¸§à¸²à¸”à¸ˆà¸¸à¸”à¸à¸¶à¹ˆà¸‡à¸à¸¥à¸²à¸‡ (à¸ªà¸µà¹€à¸‚à¸µà¸¢à¸§)
                    circle(debugFrame, Point(cx, cy), 6, Scalar(0, 255, 0), -1);
                }
            }
        }
    }

    // ðŸš€ === à¸£à¸°à¸šà¸šà¸—à¸µà¹ˆ 2: HORIZONTAL ROI (à¸”à¸±à¸à¸ˆà¸±à¸šà¸—à¸²à¸‡à¹à¸¢à¸à¸‹à¹‰à¸²à¸¢-à¸‚à¸§à¸²) === ðŸš€
    int sensorY = smoothedMask.rows * 0.7;
    int sensorHeight = 20;

    int centerWidth = smoothedMask.cols * 0.4;
    int sideWidth = (smoothedMask.cols - centerWidth) / 2;

    // à¸ªà¸£à¹‰à¸²à¸‡à¸à¸¥à¹ˆà¸­à¸‡ Rect 3 à¹ƒà¸š
    Rect leftROI(0, sensorY, sideWidth, sensorHeight);
    Rect rightROI(smoothedMask.cols - sideWidth, sensorY, sideWidth, sensorHeight);
    Rect centerROI(sideWidth, sensorY, centerWidth, sensorHeight);

    // à¸§à¸²à¸”à¸à¸¥à¹ˆà¸­à¸‡à¹€à¸‹à¸™à¹€à¸‹à¸­à¸£à¹Œà¹ƒà¸«à¹‰à¹€à¸«à¹‡à¸™à¸šà¸™à¸ˆà¸­
    rectangle(debugFrame, leftROI, Scalar(0, 165, 255), 2);
    rectangle(debugFrame, rightROI, Scalar(0, 165, 255), 2);
    rectangle(debugFrame, centerROI, Scalar(255, 0, 255), 2);

    int leftWhite = countNonZero(smoothedMask(leftROI));
    int rightWhite = countNonZero(smoothedMask(rightROI));
    int centerWhite = countNonZero(smoothedMask(centerROI));

    int sideArea = sideWidth * sensorHeight;
    int centerArea = centerWidth * sensorHeight;

    double thresholdRatio = 0.30;

    bool hasLeft = (leftWhite > sideArea * thresholdRatio);
    bool hasRight = (rightWhite > sideArea * thresholdRatio);
    bool hasCenter = (centerWhite > centerArea * thresholdRatio);

    // ðŸš€ --- à¸§à¸´à¹€à¸„à¸£à¸²à¸°à¸«à¹Œà¸›à¸£à¸°à¹€à¸ à¸—à¸—à¸²à¸‡à¹€à¸”à¸´à¸™ (à¹à¸šà¸šà¸à¸£à¸°à¸Šà¸±à¸š) --- ðŸš€
    string pathType = "UNKNOWN";

    if (hasLeft || hasRight) {
        // à¸–à¹‰à¸²à¹€à¸‹à¸™à¹€à¸‹à¸­à¸£à¹Œà¸‹à¹‰à¸²à¸¢ à¸«à¸£à¸·à¸­ à¸‚à¸§à¸² à¸—à¸³à¸‡à¸²à¸™ (à¹à¸¡à¹‰à¹€à¸žà¸µà¸¢à¸‡à¸‚à¹‰à¸²à¸‡à¹€à¸”à¸µà¸¢à¸§) à¸•à¸µà¸„à¸§à¸²à¸¡à¸§à¹ˆà¸²à¹€à¸›à¹‡à¸™ "à¸—à¸²à¸‡à¹à¸¢à¸" à¸—à¸±à¸™à¸—à¸µ
        pathType = "JUNCTION";
    }
    else if (hasCenter && !hasLeft && !hasRight) {
        // à¸–à¹‰à¸²à¸¡à¸µà¹à¸•à¹ˆà¸•à¸£à¸‡à¸à¸¥à¸²à¸‡à¹€à¸—à¹ˆà¸²à¸™à¸±à¹‰à¸™
        pathType = "STRAIGHT PATH";
    }
    else {
        // à¸–à¹‰à¸²à¹„à¸¡à¹ˆà¸¡à¸µà¸ªà¸µà¸‚à¸²à¸§à¹€à¸‚à¹‰à¸²à¹€à¸‹à¸™à¹€à¸‹à¸­à¸£à¹Œà¹€à¸¥à¸¢
        pathType = "NO PATH";
    }

    // --- à¸ªà¹ˆà¸§à¸™à¹à¸ªà¸”à¸‡à¸œà¸¥ ---
    putText(debugFrame, "Path: " + pathType, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
    string statStr = "L: " + to_string(leftWhite) + " | C: " + to_string(centerWhite) + " | R: " + to_string(rightWhite);
    putText(debugFrame, statStr, Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);

    namedWindow("1. Smoothed Mask", WINDOW_AUTOSIZE);
    namedWindow("2. Output", WINDOW_AUTOSIZE);

    imshow("1. Smoothed Mask", smoothedMask);
    imshow("2. Output", debugFrame);

    waitKey(1);

    return pathType;
}
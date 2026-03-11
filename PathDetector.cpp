
//โคดนี้
#include "PathDetector.h"
#include <vector>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

string PathDetector::detect(Mat frame)
{
    // 🚀 1. ย่อขนาดภาพให้เหลือ 640x480
    Mat workingFrame;
    resize(frame, workingFrame, Size(640, 480));

    Mat hsv, mask;
    Mat debugFrame = workingFrame.clone();

    // --- 1. ลด Noise ด้วย Gaussian Blur ---
    Mat blurredFrame;
    GaussianBlur(workingFrame, blurredFrame, Size(3, 3), 0);

    // --- 2. แปลงภาพเป็น HSV ---
    cvtColor(blurredFrame, hsv, COLOR_BGR2HSV);

    // --- 3. คัดกรองสีเพื่อสร้าง Mask ---
    Scalar lower(10, 100, 80);
    Scalar upper(35, 255, 255);
    inRange(hsv, lower, upper, mask);

    // --- 4. ลด Noise บน Mask ---
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    // --- 5. คัดแยกเฉพาะก้อนสีขาวที่ใหญ่พอ ---
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat smoothedMask = Mat::zeros(mask.size(), CV_8UC1);

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > 300) {
            drawContours(smoothedMask, contours, i, Scalar(255), FILLED);
        }
    }

    // === ลบเหลี่ยมขั้นบันไดให้เป็นก้อนเนื้อเดียวกัน ===
    GaussianBlur(smoothedMask, smoothedMask, Size(9, 9), 0);
    threshold(smoothedMask, smoothedMask, 127, 255, THRESH_BINARY);
    // ==========================================

    // 🚀 === ระบบที่ 1: SLIDING WINDOW (วาดจุดกึ่งกลางนำทาง) === 🚀
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
                    // วาดจุดกึ่งกลาง (สีเขียว)
                    circle(debugFrame, Point(cx, cy), 6, Scalar(0, 255, 0), -1);
                }
            }
        }
    }

    // 🚀 === ระบบที่ 2: HORIZONTAL ROI (ดักจับทางแยกซ้าย-ขวา) === 🚀
    int sensorY = smoothedMask.rows * 0.7;
    int sensorHeight = 20;

    int centerWidth = smoothedMask.cols * 0.4;
    int sideWidth = (smoothedMask.cols - centerWidth) / 2;

    // สร้างกล่อง Rect 3 ใบ
    Rect leftROI(0, sensorY, sideWidth, sensorHeight);
    Rect rightROI(smoothedMask.cols - sideWidth, sensorY, sideWidth, sensorHeight);
    Rect centerROI(sideWidth, sensorY, centerWidth, sensorHeight);

    // วาดกล่องเซนเซอร์ให้เห็นบนจอ
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

    // 🚀 --- วิเคราะห์ประเภททางเดิน (แบบกระชับ) --- 🚀
    string pathType = "UNKNOWN";

    if (hasLeft || hasRight) {
        // ถ้าเซนเซอร์ซ้าย หรือ ขวา ทำงาน (แม้เพียงข้างเดียว) ตีความว่าเป็น "ทางแยก" ทันที
        pathType = "JUNCTION";
    }
    else if (hasCenter && !hasLeft && !hasRight) {
        // ถ้ามีแต่ตรงกลางเท่านั้น
        pathType = "STRAIGHT PATH";
    }
    else {
        // ถ้าไม่มีสีขาวเข้าเซนเซอร์เลย
        pathType = "NO PATH";
    }

    // --- ส่วนแสดงผล ---
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
#include "pch.h"

#include "opencv2/opencv.hpp"
#include <string>

#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;


int main_motion1(std::string videoPath)
{
    Mat frame;
    Mat background;
    Mat object;

    VideoCapture cap(videoPath); 
    if (!cap.isOpened())  
        return -1;

    cap.read(frame);
    Mat acc = Mat::zeros(frame.size(), CV_32FC1);

    namedWindow("Video");
    namedWindow("Frame");
    namedWindow("Background");
    namedWindow("Foreground");

    for (;;)
    {
        Mat gray;
        cap >> frame;
        if (frame.empty()) {
            cout << "End of video." << endl;
            break; 
        }

        imshow("Video", frame);

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // running average
        // B = alpha * I + (1-alpha) * B;
        float alpha = 0.05;
        accumulateWeighted(gray, acc, alpha);

        // scale to 8-bit unsigned
        convertScaleAbs(acc, background);

        imshow("Background", background);

        // background subtraction
        // O = | I - B |
     
        GaussianBlur(gray, gray, Size(5, 5), 0);

        subtract(background, gray, object);
        imshow("Frame", object);

        bool contour_features(vector<Point> ct, Mat & img);
        threshold(object, object, 25, 255, 0);

       
        // 1. สร้างขนาด 5x5
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

        // 2. Morphological Opening หดแล้วขยาย เพื่อกำจัด Noise
        morphologyEx(object, object, MORPH_OPEN, kernel);

        // 3. Morphological Closing ขยายแล้วหด เพื่อเชื่อมส่วนที่ขาดของคนให้เป็นก้อนเดียวกัน
        morphologyEx(object, object, MORPH_CLOSE, kernel);

        vector<vector<Point>> contours;
        findContours(object, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++) {
            // 1. เช็คค HuMoments 
            if (contour_features(contours[i], frame)) {

                // 2. boundingRect 
                Rect r = boundingRect(contours[i]);

                // 3. วาดรูปสี่เหลี่ยม
                rectangle(frame, r, Scalar(0, 255, 0), 2);

            }
        }
        imshow("Video", frame); 
        

        imshow("Foreground", object);

       
        if (waitKey(30) >= 0) break; 
    }

    destroyAllWindows();
    return 0;
}
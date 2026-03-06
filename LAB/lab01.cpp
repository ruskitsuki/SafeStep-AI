#include "pch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // img
    if (argc != 2) return -1;
    Mat original = imread(argv[1]);
    if (original.empty()) return -1;

    Mat gray, binary;
    cvtColor(original, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Morphological
    // ขยายเส้นให้หนาขึ้นเพื่อเชื่อมรอยต่อ
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    morphologyEx(binary, binary, MORPH_DILATE, kernel);

    // Contours
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // เลือก Largest Radius
    int best_idx = -1;
    float max_radius = 0.0f;
    Point2f best_center;

    for (size_t i = 0; i < contours.size(); i++)
    {
        // สร้างวงกลมสมมติรอบ contour
        Point2f center;
        float radius;
        minEnclosingCircle(contours[i], center, radius);

        // กรอง Noise ทิ้ง
        if (radius < 10) continue;

        if (radius > max_radius) {
            max_radius = radius;
            best_idx = (int)i;
            best_center = center;
        }
    }

    // result
    if (best_idx != -1)
    {
        // square over 
        // ใช้ boundingRect จาก Contour
        Rect rect = boundingRect(contours[best_idx]);
        rectangle(original, rect, Scalar(0, 255, 0), 3);

        //circle over
        circle(original, best_center, (int)max_radius, Scalar(0, 0, 255), 1.5);

        string info = "Radius: " + to_string((int)max_radius);
        putText(original, info, Point(rect.x, rect.y - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    }

    namedWindow("Result", WINDOW_NORMAL);
    imshow("Result", original);

    namedWindow("Binary (Connected)", WINDOW_NORMAL);
    imshow("Binary (Connected)", binary);

    waitKey(0);
    return 0;
}
#include <opencv2/opencv.hpp>
#include "PathDetector.h"

using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap("E:/AiE202/SafeStep-AI/Dataset/IMG_6498.mp4"); // VDO file path
    

    if (!cap.isOpened())
    {
        cout << "Cannot open video" << endl;
        return -1;
    }

    PathDetector detector;

    Mat frame;

    while (cap.read(frame))
    {
        string direction = detector.detect(frame);

        putText(frame, direction, Point(50, 50),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

        imshow("Path Tracking", frame);

        if (waitKey(30) == 27)
            break;
    }

    return 0;
}


//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//int main()
//{
//    // เปิดกล้อง (หรือใส่ path ของวิดีโอแทน 0)
//    VideoCapture cap("S:/Pictures/Screenshots/Screenshot 2026 - 03 - 09 054720.png"");
//    if (!cap.isOpened()) return -1;
//
//    // ตัวแปรสำหรับเก็บค่า HSV
//    int hMin = 0, sMin = 0, vMin = 0;
//    int hMax = 179, sMax = 255, vMax = 255;
//
//    // สร้างหน้าต่างสำหรับแสดงแถบเลื่อน
//    namedWindow("Trackbars", WINDOW_NORMAL);
//    resizeWindow("Trackbars", 400, 300);
//
//    // สร้าง Trackbar 6 ตัว (Min และ Max สำหรับ H, S, V)
//    // หมายเหตุ: ใน OpenCV ค่า Hue มีช่วง 0-179 ส่วน Saturation และ Value มีช่วง 0-255
//    createTrackbar("Hue Min", "Trackbars", &hMin, 179);
//    createTrackbar("Hue Max", "Trackbars", &hMax, 179);
//    createTrackbar("Sat Min", "Trackbars", &sMin, 255);
//    createTrackbar("Sat Max", "Trackbars", &sMax, 255);
//    createTrackbar("Val Min", "Trackbars", &vMin, 255);
//    createTrackbar("Val Max", "Trackbars", &vMax, 255);
//
//    Mat frame, hsv, mask;
//
//    while (true)
//    {
//        cap.read(frame);
//        if (frame.empty()) break;
//
//        // แปลงเป็น HSV
//        cvtColor(frame, hsv, COLOR_BGR2HSV);
//
//        // ดึงค่าจาก Trackbar มาสร้างขอบเขตสี
//        Scalar lower(hMin, sMin, vMin);
//        Scalar upper(hMax, sMax, vMax);
//
//        // สร้าง Mask
//        inRange(hsv, lower, upper, mask);
//
//        // แสดงผล
//        imshow("Original", frame);
//        imshow("Mask", mask);
//
//        // กด 'q' เพื่อออก
//        if (waitKey(1) == 'q') break;
//    }
//
//    // ปริ้นค่าที่จูนได้ออกมา เพื่อเอาไปใส่ในโค้ดหลัก
//    cout << "--- Your HSV Values ---" << endl;
//    cout << "Scalar lower(" << hMin << ", " << sMin << ", " << vMin << ");" << endl;
//    cout << "Scalar upper(" << hMax << ", " << sMax << ", " << vMax << ");" << endl;
//
//    return 0;
//}
#include "pch.h"
#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>

using namespace cv;
using namespace std;


bool contour_features(vector<Point> ct, Mat& img);

int main_motion2(std::string videoPath)
{
    Mat frame, fgMaskMOG2;
    Ptr<BackgroundSubtractor> pMOG2;

    VideoCapture cap(videoPath);
    if (!cap.isOpened()) return -1;

    // 1. สร้าง MOG2 
    pMOG2 = createBackgroundSubtractorMOG2();

    namedWindow("frame", 1);
    namedWindow("FG Mask MOG 2");

    for (;;)
    {
        cap >> frame;
        if (frame.empty()) {
            cout << "End of video." << endl;
            break; 
        }

        // 2. ลด Noise เบื้องต้นก่อนส่งเข้าโมเดล MOG2
        Mat blurred;
        GaussianBlur(frame, blurred, Size(5, 5), 0);

        // 3. ใช้ MOG2 แยก Foreground ออกจาก Background
        pMOG2->apply(blurred, fgMaskMOG2);

        // 4. Threshold
        // MOG2 มักจะให้ค่าเงาเป็นสีเทา (ประมาณ 127) เราจึงตัดออกโดยใช้ค่า 220
        threshold(fgMaskMOG2, fgMaskMOG2, 220, 255, THRESH_BINARY);

        // 5. Morphological Operations เชื่อมก้อน
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_OPEN, kernel);
        morphologyEx(fgMaskMOG2, fgMaskMOG2, MORPH_CLOSE, kernel);

        // 6. ค้นหา Contours
        vector<vector<Point>> contours;
        findContours(fgMaskMOG2, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++) {
            // 1. เช็คค HuMoments
            if (contour_features(contours[i], frame)) {

                // 2. boundingRect 
                Rect r = boundingRect(contours[i]);

                // 3. วาดรูปสี่เหลี่ยม
                rectangle(frame, r, Scalar(0, 255, 0), 2);
            }
        }

        imshow("frame", frame);
        imshow("FG Mask MOG 2", fgMaskMOG2);

        if (waitKey(30) >= 0) break;
    }

    destroyAllWindows();
    return 0;
}
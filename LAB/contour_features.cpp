#include "pch.h"
#include "opencv2/opencv.hpp"
#include <string>

using namespace cv;
using namespace std;

bool contour_features(vector<Point> ct, Mat& img)
{
    // 1. ตรวจสอบจำนวนจุด
    if (ct.size() < 5) return false;

    // 2. ตรวจสอบพื้นที่ กรองวัตถุที่เล็กเกินไปออก
    double area = contourArea(ct);
    if (area < 500) return false;

    // 3. คำนวณ Moments เพื่อหาจุดศูนย์กลาง
    Moments mu = moments(ct);
    if (mu.m00 == 0) return false; 
    Point center(mu.m10 / mu.m00, mu.m01 / mu.m00);

    // 4. HuMoments
    double hu[7];
    HuMoments(mu, hu);

    // 5. แสดงค่า Hu[0] บนภาพตรงตำแหน่งวัตถุ
    string text = "Hu0: " + to_string(hu[0]).substr(0, 6);
    putText(img, text, center, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);

    // 6. เงื่อนไขตัดสินว่าเป็นคน
    if (hu[0] < 0.8) return true;

    return false;
}
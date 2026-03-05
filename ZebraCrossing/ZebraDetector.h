// ============================================================
//  ZebraDetector.h
//  Header file สำหรับคลาส ZebraDetector
//
//  ใช้งาน: รวมไฟล์นี้ในโปรเจกต์ของเพื่อน แล้วเรียก
//          detector.DetectAndDraw(frame);
// ============================================================
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>

class ZebraDetector {
public:
    // ---- Constructor / Setup ----

    // สร้าง detector พร้อมโหลดโมเดลทันที
    // modelPath: path ไปยัง ZebraModel.xml
    explicit ZebraDetector(const std::string& modelPath);

    // ---- Main API (ส่งมอบให้ทีม) ----

    // ฟังก์ชันหลัก: ตรวจจับทางม้าลายในเฟรม แล้ววาดกรอบสีเขียว
    // frame : BGR image (จากกล้องหรือวิดีโอ)
    // คืนค่า: true ถ้าพบทางม้าลายในเฟรมนี้
    bool DetectAndDraw(cv::Mat& frame);

    // ---- Utility ----

    // โหลดโมเดลจากไฟล์ (ถ้ายังไม่ได้โหลด หรือต้องการเปลี่ยนโมเดล)
    bool loadModel(const std::string& modelPath);

    // ตรวจสอบว่าโมเดลโหลดสำเร็จหรือยัง
    bool isReady() const { return m_ann && m_modelLoaded; }

    // ---- Tuning Parameters (ปรับได้) ----

    // ขนาดภาพที่ส่งเข้า ANN (ต้องตรงกับที่ train ไว้)
    static const int FEATURE_SIZE = 32;

    // พื้นที่ contour ขั้นต่ำ (กรอง noise เล็กๆ ออก)
    double minContourArea  = 3000.0;

    // Aspect ratio ของ bounding box ที่ยอมรับ (กว้าง/สูง)
    double minAspectRatio  = 0.8;
    double maxAspectRatio  = 10.0;

    // Threshold ของ ANN: ถ้า output[0] > threshold → ถือว่าเป็นทางม้าลาย
    float  annThreshold    = 0.4f;

    // สีกรอบ (BGR)
    cv::Scalar boxColor    = cv::Scalar(0, 255, 0);   // สีเขียว
    int        boxThick    = 3;

private:
    // ---- Internal Pipeline ----

    // แปลงเฟรมเป็น binary mask ของพื้นที่สีขาว
    cv::Mat preprocess(const cv::Mat& frame);

    // ตัดภาพ ROI → resize → flatten → normalize เป็น feature vector
    cv::Mat extractFeature(const cv::Mat& gray, const cv::Rect& roi);

    // ส่ง feature vector เข้า ANN → คืนค่า score ของ class "zebra"
    float classify(const cv::Mat& feature);

    // ---- Members ----
    cv::Ptr<cv::ml::ANN_MLP> m_ann;
    bool m_modelLoaded = false;
};

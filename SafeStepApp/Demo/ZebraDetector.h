#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <deque>

class ZebraDetector
{
private:
    cv::Ptr<cv::ml::ANN_MLP> model;

    // [FIX 4] Temporal Smoothing: เก็บประวัติการ detect N frame ล่าสุด
    // sentinel Rect(0,0,0,0) = frame ที่ไม่พบทางม้าลาย
    std::deque<cv::Rect> detectionHistory;
    int historySize    = 7;  // จำนวน frame ที่เก็บไว้ (ปรับได้: 5-10)
    int minDetectCount = 3;  // ต้องพบ >= 3 ครั้งจาก 7 frame จึงแสดงกรอบ



    // [DEBUG] แสดงหน้าต่าง intermediate แต่ละขั้นตอน
    bool debugMode = false;

    // สกัด Feature 160 ค่า (120 col-sum + 40 row-sum) ตรงกับโมเดลที่ retrain แล้ว
    cv::Mat ExtractFeature(const cv::Mat& cropImage);

    // รันด่านกรองซี่ทั้ง 4 ด่าน + ANN predict → คืน true ถ้าพบทางม้าลาย
    bool TryDetectZebra(const cv::Mat&              workFrame,
                        const std::vector<cv::Rect>& validStripes,
                        cv::Rect&                    outRect);

public:
    ZebraDetector();
    ~ZebraDetector();

    // โหลดโมเดล (ZebraModel.xml ที่ retrain ด้วย 160 features แล้ว)
    bool LoadModel(const std::string& modelPath);

    // ค้นหาและวาดกรอบสีเขียวรอบทางม้าลายในแต่ละเฟรม
    void DetectAndDraw(cv::Mat& frame);



    // [DEBUG] เปิด/ปิดหน้าต่าง debug (กด D ในหน้าต่าง Demo)
    void SetDebugMode(bool enabled);
    bool IsDebugMode()        const { return debugMode; }
};

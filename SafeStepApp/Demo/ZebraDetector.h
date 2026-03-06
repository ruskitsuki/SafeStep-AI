#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class ZebraDetector
{
private:
    cv::Ptr<cv::ml::ANN_MLP> model;
    
    // ฟังก์ชันย่อยสำหรับสกัด Feature จาก Bounding Box ที่ครอบได้ (อิง LprFeature)
    cv::Mat ExtractFeature(cv::Mat& cropImage);

public:
    ZebraDetector();
    ~ZebraDetector();
    
    // โหลดโมเดลที่ผ่านการสอนมาแล้ว (เช่น ZebraModel.xml)
    bool LoadModel(const std::string& modelPath);
    
    // ค้นหาและวาดกรอบสีเขียวรอบทางม้าลายในแต่ละเฟรม 
    void DetectAndDraw(cv::Mat& frame);
};

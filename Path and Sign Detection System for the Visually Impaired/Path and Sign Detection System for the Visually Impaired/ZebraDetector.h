#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <deque>

class ZebraDetector
{
private:
    cv::Ptr<cv::ml::ANN_MLP> model;

    // Temporal Smoothing: à¹€à¸à¹‡à¸šà¸›à¸£à¸°à¸§à¸±à¸•à¸´à¸à¸²à¸£ detect N frame à¸¥à¹ˆà¸²à¸ªà¸¸à¸”
    // sentinel Rect(0,0,0,0) = frame à¸—à¸µà¹ˆà¹„à¸¡à¹ˆà¸žà¸šà¸—à¸²à¸‡à¸¡à¹‰à¸²à¸¥à¸²à¸¢
    std::deque<cv::Rect> detectionHistory;
    int historySize = 7;  // à¸ˆà¸³à¸™à¸§à¸™ frame à¸—à¸µà¹ˆà¹€à¸à¹‡à¸šà¹„à¸§à¹‰ (à¸›à¸£à¸±à¸šà¹„à¸”à¹‰: 5-10)
    int minDetectCount = 3;  // à¸•à¹‰à¸­à¸‡à¸žà¸š >= 3 à¸„à¸£à¸±à¹‰à¸‡à¸ˆà¸²à¸ 7 frame à¸ˆà¸¶à¸‡à¹à¸ªà¸”à¸‡à¸à¸£à¸­à¸š



    // [DEBUG] à¹à¸ªà¸”à¸‡à¸«à¸™à¹‰à¸²à¸•à¹ˆà¸²à¸‡ intermediate à¹à¸•à¹ˆà¸¥à¸°à¸‚à¸±à¹‰à¸™à¸•à¸­à¸™
    bool debugMode = false;

    // à¸ªà¸à¸±à¸” Feature 160 à¸„à¹ˆà¸² (120 col-sum + 40 row-sum) à¸•à¸£à¸‡à¸à¸±à¸šà¹‚à¸¡à¹€à¸”à¸¥à¸—à¸µà¹ˆ retrain à¹à¸¥à¹‰à¸§
    cv::Mat ExtractFeature(const cv::Mat& cropImage);

    // à¸£à¸±à¸™à¸à¸²à¸£à¸à¸£à¸­à¸‡à¸‹à¸µà¹ˆà¸—à¸±à¹‰à¸‡ 4 à¸”à¹ˆà¸²à¸™ + ANN predict â†’ à¸„à¸·à¸™ true à¸–à¹‰à¸²à¸žà¸šà¸—à¸²à¸‡à¸¡à¹‰à¸²à¸¥à¸²à¸¢
    bool TryDetectZebra(const cv::Mat& workFrame,
        const std::vector<cv::Rect>& validStripes,
        cv::Rect& outRect);

public:
    ZebraDetector();
    ~ZebraDetector();

    // à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥ (ZebraModel.xml à¸—à¸µà¹ˆ retrain à¸”à¹‰à¸§à¸¢ 160 features à¹à¸¥à¹‰à¸§)
    bool LoadModel(const std::string& modelPath);

    // à¸„à¹‰à¸™à¸«à¸²à¹à¸¥à¸°à¸§à¸²à¸”à¸à¸£à¸­à¸šà¸ªà¸µà¹€à¸‚à¸µà¸¢à¸§à¸£à¸­à¸šà¸—à¸²à¸‡à¸¡à¹‰à¸²à¸¥à¸²à¸¢à¹ƒà¸™à¹à¸•à¹ˆà¸¥à¸°à¹€à¸Ÿà¸£à¸¡
    bool DetectAndDraw(cv::Mat& frame);



    // à¹€à¸›à¸´à¸”/à¸›à¸´à¸”à¸«à¸™à¹‰à¸²à¸•à¹ˆà¸²à¸‡ debug (à¸à¸” D à¹ƒà¸™à¸«à¸™à¹‰à¸²à¸•à¹ˆà¸²à¸‡ Demo)
    void SetDebugMode(bool enabled);
    bool IsDebugMode()        const { return debugMode; }
};
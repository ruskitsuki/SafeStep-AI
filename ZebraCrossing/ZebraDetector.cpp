#include "pch.h"
// ============================================================
//  ZebraDetector.cpp
//  Implementation ของ ZebraDetector
//
//  Pipeline:
//   frame → Grayscale → AdaptiveThreshold → Morphology
//        → findContours → กรอง ROI → ANN classify → วาดกรอบ
// ============================================================
#include "ZebraDetector.h"
#include <iostream>

using namespace cv;
using namespace std;

// ================================================================
//  Constructor
// ================================================================
ZebraDetector::ZebraDetector(const std::string& modelPath) {
    if (!loadModel(modelPath)) {
        cerr << "[ZebraDetector] WARNING: โหลดโมเดลไม่สำเร็จ: "
             << modelPath << endl;
    }
}

// ================================================================
//  loadModel
// ================================================================
bool ZebraDetector::loadModel(const std::string& modelPath) {
    try {
        m_ann = cv::ml::ANN_MLP::load(modelPath);
        if (m_ann.empty()) {
            m_modelLoaded = false;
            return false;
        }
        m_modelLoaded = true;
        cout << "[ZebraDetector] โหลดโมเดลสำเร็จ: " << modelPath << endl;
        return true;
    }
    catch (const cv::Exception& e) {
        cerr << "[ZebraDetector] Exception ขณะโหลดโมเดล: " << e.what() << endl;
        m_modelLoaded = false;
        return false;
    }
}

// ================================================================
//  preprocess
//  แปลง frame → binary mask เน้นพื้นที่สีขาว
// ================================================================
Mat ZebraDetector::preprocess(const Mat& frame) {
    Mat gray, blurred, binary, morph;

    // 1. Grayscale
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // 2. Gaussian Blur เพื่อลด noise
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // 3. Adaptive Threshold
    //    ใช้ ADAPTIVE แทน global threshold เพื่อรองรับแสงที่เปลี่ยนแปลง
    adaptiveThreshold(blurred, binary,
                      255,
                      ADAPTIVE_THRESH_GAUSSIAN_C,
                      THRESH_BINARY,
                      15,   // blockSize (ต้องเป็นเลขคี่ > 1)
                      -5);  // C: ลบออกจาก mean (ค่าลบ = เน้นพื้นสว่าง)

    // 4. Morphological Closing
    //    เชื่อมแถบสีขาวของทางม้าลายหลายเส้นให้กลายเป็น blob เดียว
    Mat kernel = getStructuringElement(MORPH_RECT, Size(20, 10));
    morphologyEx(binary, morph, MORPH_CLOSE, kernel);

    // 5. Dilation เพิ่มเติมเพื่อขยาย blob ให้ชัดขึ้น
    Mat dilateKernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(morph, morph, dilateKernel, Point(-1,-1), 2);

    return morph;
}

// ================================================================
//  extractFeature
//  ตัด ROI → Grayscale resize → flatten → normalize [0,1]
// ================================================================
Mat ZebraDetector::extractFeature(const Mat& gray, const Rect& roi) {
    // ป้องกัน ROI เกินขอบภาพ
    Rect safeRoi = roi & Rect(0, 0, gray.cols, gray.rows);
    Mat cropped  = gray(safeRoi);

    // Resize เป็น FEATURE_SIZE x FEATURE_SIZE
    Mat resized;
    resize(cropped, resized, Size(FEATURE_SIZE, FEATURE_SIZE));

    // Flatten → 1D row vector
    Mat flat = resized.reshape(1, 1);

    // Normalize เป็น float [0.0, 1.0]
    Mat feature;
    flat.convertTo(feature, CV_32F, 1.0 / 255.0);

    return feature;  // size: 1 x (FEATURE_SIZE*FEATURE_SIZE)
}

// ================================================================
//  classify
//  ส่ง feature เข้า ANN → คืนค่า score ของ class 0 (zebra)
// ================================================================
float ZebraDetector::classify(const Mat& feature) {
    if (!m_modelLoaded) return 0.0f;

    Mat output;
    m_ann->predict(feature, output);
    // output: 1 x 2  [score_zebra, score_road]
    return output.at<float>(0, 0);
}

// ================================================================
//  DetectAndDraw  (ฟังก์ชันหลักที่ส่งมอบให้ทีม)
// ================================================================
bool ZebraDetector::DetectAndDraw(Mat& frame) {
    if (frame.empty()) return false;

    // --- Pre-process ---
    Mat mask = preprocess(frame);

    // Grayscale ของ frame เดิม (ใช้ extract feature)
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // --- Find Contours ---
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    bool foundZebra = false;

    for (const auto& contour : contours) {
        // กรองพื้นที่เล็กเกินไป
        double area = contourArea(contour);
        if (area < minContourArea) continue;

        // Bounding Box
        Rect bbox = boundingRect(contour);

        // กรอง Aspect Ratio (ทางม้าลายมักกว้างกว่าสูง)
        double aspectRatio = static_cast<double>(bbox.width) / bbox.height;
        if (aspectRatio < minAspectRatio || aspectRatio > maxAspectRatio)
            continue;

        // --- Extract Feature & Classify ---
        Mat feature = extractFeature(gray, bbox);
        float score = classify(feature);

        // ถ้า score ผ่าน threshold → ถือว่าเป็นทางม้าลาย
        if (score > annThreshold) {
            // วาดกรอบสีเขียวบน frame
            rectangle(frame, bbox, boxColor, boxThick);

            // ใส่ข้อความ label
            string label = "Zebra (" + to_string((int)(score * 100)) + "%)";
            int    baseLine = 0;
            Size   textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX,
                                          0.7, 2, &baseLine);
            Point  textOrg(bbox.x, bbox.y - 10);
            if (textOrg.y < 10) textOrg.y = bbox.y + textSize.height + 10;

            // พื้นหลังข้อความ
            rectangle(frame,
                      textOrg + Point(0, baseLine),
                      textOrg + Point(textSize.width, -textSize.height),
                      Scalar(0, 0, 0), FILLED);
            putText(frame, label, textOrg,
                    FONT_HERSHEY_SIMPLEX, 0.7, boxColor, 2);

            foundZebra = true;
        }
    }

    return foundZebra;
}

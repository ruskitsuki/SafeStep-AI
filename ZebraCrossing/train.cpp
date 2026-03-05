#include "pch.h"
// ============================================================
//  train.cpp
//  โปรแกรมสำหรับ Train ANN โมเดลและ Export เป็น ZebraModel.xml
//
//  วิธีใช้งาน:
//   1. รัน main_augment.cpp เพื่อสร้าง dataset/ ก่อน
//   2. รัน train.cpp → จะสร้าง ZebraModel.xml
//   3. นำ ZebraModel.xml ไปใช้กับ ZebraDetector
// ============================================================
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// ---- Parameters ----
static const int   FEATURE_SIZE  = 32;       // ขนาดภาพ input (32x32)
static const int   INPUT_NODES   = FEATURE_SIZE * FEATURE_SIZE; // 1024
static const int   HIDDEN_NODES  = 64;       // ปรับได้ (ลอง 32, 64, 128)
static const int   OUTPUT_NODES  = 2;        // zebra=0, road=1
static const int   MAX_ITER      = 3000;     // รอบ training สูงสุด
static const double EPSILON      = 0.0001;   // เกณฑ์หยุดเทรน
static const string MODEL_PATH   = "ZebraModel.xml";
// label ของแต่ละคลาส
// zebra → [1, 0]
// road  → [0, 1]
// -------------------

// ================================================================
//  โหลดรูปภาพจากโฟลเดอร์ → แปลงเป็น feature vector + label
// ================================================================
void loadClass(const string& dirPath,
               float labelZebra,   // 1.0f สำหรับ zebra, 0.0f สำหรับ road
               float labelRoad,    // 0.0f สำหรับ zebra, 1.0f สำหรับ road
               vector<Mat>& features,
               vector<Mat>& labels) {

    if (!fs::exists(dirPath)) {
        cerr << "[ERROR] ไม่พบโฟลเดอร์: " << dirPath << endl;
        return;
    }

    int loaded = 0;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        string ext = entry.path().extension().string();
        // รองรับนามสกุล .jpg .png .bmp
        if (ext != ".jpg" && ext != ".png" && ext != ".bmp") continue;

        // โหลดภาพ
        Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        // Resize → FEATURE_SIZE x FEATURE_SIZE
        Mat resized;
        resize(img, resized, Size(FEATURE_SIZE, FEATURE_SIZE));

        // Flatten + Normalize [0, 1]
        Mat flat = resized.reshape(1, 1);
        Mat feature;
        flat.convertTo(feature, CV_32F, 1.0 / 255.0);  // 1 x 1024

        // Label row: [labelZebra, labelRoad]
        Mat label = (Mat_<float>(1, 2) << labelZebra, labelRoad);

        features.push_back(feature);
        labels.push_back(label);
        loaded++;
    }
    cout << "[INFO] โหลดจาก " << dirPath << " : " << loaded << " รูป" << endl;
}

// ================================================================
//  สร้างและ Train ANN
// ================================================================
int main() {
    cout << "=====================================" << endl;
    cout << "   Zebra Crossing ANN Training       " << endl;
    cout << "=====================================" << endl;

    // ---- 1. โหลด Dataset ----
    vector<Mat> features, labels;

    // class 0: ทางม้าลาย  → label [1, 0]
    loadClass("dataset/zebra", 1.0f, 0.0f, features, labels);

    // class 1: ถนนปกติ    → label [0, 1]
    loadClass("dataset/road",  0.0f, 1.0f, features, labels);

    if (features.empty()) {
        cerr << "[ERROR] ไม่มีข้อมูล กรุณารัน main_augment.cpp ก่อน" << endl;
        return -1;
    }
    cout << "[INFO] รวมข้อมูลทั้งหมด: " << features.size() << " รูป" << endl;

    // ---- 2. Shuffle ข้อมูล (สุ่มลำดับก่อน train) ----
    vector<int> indices(features.size());
    iota(indices.begin(), indices.end(), 0);
    mt19937 rng(42);
    shuffle(indices.begin(), indices.end(), rng);

    // ---- 3. รวม Mat เป็น Matrix ใหญ่ (N x 1024) และ (N x 2) ----
    Mat trainData(static_cast<int>(features.size()), INPUT_NODES,  CV_32F);
    Mat trainLabels(static_cast<int>(features.size()), OUTPUT_NODES, CV_32F);

    for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        features[indices[i]].copyTo(trainData.row(i));
        labels[indices[i]].copyTo(trainLabels.row(i));
    }

    // ---- 4. แบ่ง Train / Validation (80:20) ----
    int totalSamples = trainData.rows;
    int trainCount   = static_cast<int>(totalSamples * 0.8);

    Mat X_train = trainData.rowRange(0, trainCount);
    Mat Y_train = trainLabels.rowRange(0, trainCount);
    Mat X_val   = trainData.rowRange(trainCount, totalSamples);
    Mat Y_val   = trainLabels.rowRange(trainCount, totalSamples);

    cout << "[INFO] Train samples: " << X_train.rows
         << "  Validation samples: " << X_val.rows << endl;

    // ---- 5. สร้าง ANN_MLP ----
    auto ann = cv::ml::ANN_MLP::create();

    // กำหนดโครงสร้าง: Input → Hidden → Output
    Mat layerSizes = (Mat_<int>(1, 3) << INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES);
    ann->setLayerSizes(layerSizes);

    // Activation function: SIGMOID_SYM (range -1 ถึง 1)
    ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);

    // Termination criteria
    ann->setTermCriteria(TermCriteria(
        TermCriteria::MAX_ITER + TermCriteria::EPS,
        MAX_ITER,
        EPSILON
    ));

    // Training algorithm: RPROP (robust backpropagation, แนะนำ)
    ann->setTrainMethod(cv::ml::ANN_MLP::RPROP);

    // ---- 6. เตรียม TrainData ----
    Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(
        X_train, cv::ml::ROW_SAMPLE, Y_train
    );

    // ---- 7. Train ----
    cout << "\n[INFO] เริ่ม Training... (อาจใช้เวลา 1-5 นาที)" << endl;
    ann->train(td);
    cout << "[INFO] Training เสร็จสิ้น!" << endl;

    // ---- 8. Evaluate บน Validation set ----
    cout << "\n[INFO] ประเมินผลบน Validation set..." << endl;
    Mat prediction;
    ann->predict(X_val, prediction);

    int correct = 0;
    for (int i = 0; i < X_val.rows; i++) {
        // หา class ที่ predict (index ของค่าสูงสุด)
        float p0 = prediction.at<float>(i, 0);  // score zebra
        float p1 = prediction.at<float>(i, 1);  // score road
        int predictedClass = (p0 > p1) ? 0 : 1;

        float l0 = Y_val.at<float>(i, 0);
        float l1 = Y_val.at<float>(i, 1);
        int   trueClass = (l0 > l1) ? 0 : 1;

        if (predictedClass == trueClass) correct++;
    }

    float accuracy = (X_val.rows > 0)
                     ? (float)correct / X_val.rows * 100.f
                     : 0.f;
    cout << "[RESULT] Validation Accuracy: " << accuracy
         << "% (" << correct << "/" << X_val.rows << ")" << endl;

    // ---- 9. Save Model ----
    ann->save(MODEL_PATH);
    cout << "\n[DONE] บันทึกโมเดลเป็น: " << MODEL_PATH << endl;

    if (accuracy < 70.f) {
        cout << "\n[WARNING] Accuracy ต่ำกว่า 70%" << endl;
        cout << "  แนะนำ: เพิ่มจำนวนรูปใน dataset หรือปรับ HIDDEN_NODES" << endl;
    } else {
        cout << "[INFO] โมเดลพร้อมใช้งาน!" << endl;
    }

    cout << "\nกด Enter เพื่อออก...";
    cin.get();
    return 0;
}

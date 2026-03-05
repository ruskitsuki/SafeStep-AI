#include "pch.h"
// ============================================================
//  main_augment.cpp
//  Data Augmentation Script สำหรับสร้าง Dataset ทางม้าลาย
//
//  วิธีใช้งาน:
//   1. วางภาพต้นแบบทางม้าลาย 1 รูป  ชื่อ: zebra_src.jpg
//      วางภาพต้นแบบถนนปกติ    1 รูป  ชื่อ: road_src.jpg
//      ทั้งสองไฟล์อยู่ที่โฟลเดอร์เดียวกับ exe
//   2. รันโปรแกรม → จะสร้างโฟลเดอร์
//         dataset/zebra/   (100 รูป)
//         dataset/road/    (100 รูป)
// ============================================================
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// ---- Parameters ----
static const int   NUM_IMAGES   = 100;   // จำนวนรูปที่สร้างต่อ class
static const int   OUTPUT_SIZE  = 32;    // ขนาด output ที่ resize (ใช้กับ ANN)
static const float ROT_RANGE    = 15.0f; // องศาสูงสุดในการหมุน ±
static const float BRIGHT_ALPHA_MIN = 0.7f;  // ความสว่าง (contrast) ต่ำสุด
static const float BRIGHT_ALPHA_MAX = 1.3f;  // ความสว่าง (contrast) สูงสุด
static const float BRIGHT_BETA_MIN  = -40.f; // brightness offset ต่ำสุด
static const float BRIGHT_BETA_MAX  =  40.f; // brightness offset สูงสุด
// -------------------

// ฟังก์ชัน: สุ่มตัวเลือก float ในช่วง [lo, hi]
static float randFloat(mt19937& rng, float lo, float hi) {
    uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

// ฟังก์ชัน: augment รูปภาพ 1 รูป
Mat augmentImage(const Mat& src, mt19937& rng) {
    Mat result = src.clone();

    // 1. Random Rotation
    float angle = randFloat(rng, -ROT_RANGE, ROT_RANGE);
    Point2f center(result.cols / 2.0f, result.rows / 2.0f);
    Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(result, result, rotMat, result.size(),
               INTER_LINEAR, BORDER_REFLECT);

    // 2. Random Brightness & Contrast  →  output = input * alpha + beta
    float alpha = randFloat(rng, BRIGHT_ALPHA_MIN, BRIGHT_ALPHA_MAX);
    float beta  = randFloat(rng, BRIGHT_BETA_MIN,  BRIGHT_BETA_MAX);
    result.convertTo(result, -1, alpha, beta);

    // 3. Random Gaussian Blur (จำลองการสั่นของกล้อง)
    //    kernel size ต้องเป็นเลขคี่ → สุ่ม 1, 3, 5
    static const int kernels[] = {1, 3, 5};
    uniform_int_distribution<int> kDist(0, 2);
    int ksize = kernels[kDist(rng)];
    if (ksize > 1) {
        GaussianBlur(result, result, Size(ksize, ksize), 0);
    }

    // 4. Random Flip แนวนอน (50%)
    uniform_int_distribution<int> flipDist(0, 1);
    if (flipDist(rng)) {
        flip(result, result, 1);
    }

    // 5. Resize เป็น OUTPUT_SIZE x OUTPUT_SIZE
    resize(result, result, Size(OUTPUT_SIZE, OUTPUT_SIZE));

    return result;
}

// ฟังก์ชัน: สร้างรูปภาพ augmented สำหรับคลาสหนึ่ง
void generateDataset(const string& srcPath,
                     const string& outDir,
                     const string& prefix,
                     mt19937& rng) {
    // โหลดรูปต้นแบบ
    Mat src = imread(srcPath);
    if (src.empty()) {
        cerr << "[ERROR] ไม่พบไฟล์: " << srcPath << endl;
        exit(1);
    }
    cout << "[INFO] โหลด " << srcPath
         << "  (" << src.cols << "x" << src.rows << ")" << endl;

    // สร้างโฟลเดอร์ output
    fs::create_directories(outDir);

    int created = 0;
    while (created < NUM_IMAGES) {
        Mat aug = augmentImage(src, rng);

        // ตั้งชื่อไฟล์ เช่น zebra_001.jpg
        char filename[64];
        snprintf(filename, sizeof(filename), "%s_%03d.jpg",
                 prefix.c_str(), created + 1);
        string outPath = outDir + "/" + filename;

        if (!imwrite(outPath, aug)) {
            cerr << "[ERROR] บันทึกไฟล์ไม่ได้: " << outPath << endl;
        }
        created++;
    }
    cout << "[INFO] สร้าง " << created << " รูป → " << outDir << endl;
}

int main() {
    cout << "=====================================" << endl;
    cout << "  Zebra Crossing Data Augmentation   " << endl;
    cout << "=====================================" << endl;

    // Random number generator (seed ด้วย random_device)
    random_device rd;
    mt19937 rng(rd());

    // สร้าง dataset สำหรับ "ทางม้าลาย"
    generateDataset("zebra_src.jpg",
                    "dataset/zebra",
                    "zebra",
                    rng);

    // สร้าง dataset สำหรับ "ถนนปกติ"
    generateDataset("road_src.jpg",
                    "dataset/road",
                    "road",
                    rng);

    cout << "\n[DONE] สร้าง Dataset เสร็จสิ้น!" << endl;
    cout << "  - dataset/zebra/  : " << NUM_IMAGES << " รูป" << endl;
    cout << "  - dataset/road/   : " << NUM_IMAGES << " รูป" << endl;
    cout << "\nกด Enter เพื่อออก...";
    cin.get();
    return 0;
}

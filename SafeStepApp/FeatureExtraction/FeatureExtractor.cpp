#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// ===========================================================================
// [FIX 6] สกัด Feature 160 ค่า: 120 col-sum + 40 row-sum
// (เหมือน ZebraDetector::ExtractFeature แต่เขียนลงไฟล์ CSV แทน)
// ===========================================================================
void gen_feature_input(const Mat& image, const string& output_filename)
{
    Mat greyMat;
    if (image.channels() > 1)
        cvtColor(image, greyMat, COLOR_BGR2GRAY);
    else
        greyMat = image.clone();

    // --- Column sums (120 values) ---
    float max_col = 1.0f;
    vector<int> col_sums(greyMat.cols, 0);
    for (int i = 0; i < greyMat.cols; i++) {
        for (int k = 0; k < greyMat.rows; k++)
            col_sums[i] += greyMat.at<unsigned char>(k, i);
        if ((float)col_sums[i] > max_col) max_col = (float)col_sums[i];
    }

    // --- Row sums (40 values) ---
    float max_row = 1.0f;
    vector<int> row_sums(greyMat.rows, 0);
    for (int k = 0; k < greyMat.rows; k++) {
        for (int i = 0; i < greyMat.cols; i++)
            row_sums[k] += greyMat.at<unsigned char>(k, i);
        if ((float)row_sums[k] > max_row) max_row = (float)row_sums[k];
    }

    // บันทึก: col_sums normalize ก่อน แล้วตามด้วย row_sums normalize
    ofstream myfile(output_filename, ios::app);
    for (int i = 0; i < greyMat.cols; i++)
        myfile << ((float)col_sums[i] / max_col) << ",";
    for (int k = 0; k < greyMat.rows; k++)
        myfile << ((float)row_sums[k] / max_row) << ",";
    myfile << "\n";
    myfile.close();
}

// ===========================================================================
// Label (One-hot encoding): 0 = non-zebra, 1 = zebra
// ===========================================================================
void gen_feature_output(const string& output_filename, int out, int num = 2)
{
    ofstream myfile(output_filename, ios::app);
    for (int i = 0; i < num; i++)
        myfile << (i == out ? 1 : 0) << ",";
    myfile << "\n";
    myfile.close();
}

// ===========================================================================
int main(int argc, char** argv)
{
    string images_dir_pos = "../images/";
    string images_dir_neg = "../images_non_zebra/";
    string feature_file   = "features.txt";
    string label_file     = "labels.txt";

    // ล้างไฟล์เก่า
    ofstream(feature_file, ios::trunc).close();
    ofstream(label_file,   ios::trunc).close();

    cout << "--- SafeStep-AI Feature Extraction ---\n";
    cout << "Output: 160 features per image (120 col-sum + 40 row-sum)\n\n";

    int success_count = 0;
    Size fixedSize(120, 40);  // resize ก่อนสกัด feature (เหมือน ZebraDetector)

    // ─────────── Positive (ทางม้าลาย: Label = 1) ───────────
    cout << "Reading POSITIVE images from: " << images_dir_pos << "\n";
    if (fs::exists(images_dir_pos)) {
        for (const auto& entry : fs::directory_iterator(images_dir_pos)) {
            string fp = entry.path().string();
            if (fp.find(".jpg") == string::npos &&
                fp.find(".png") == string::npos) continue;

            Mat src = imread(fp, IMREAD_COLOR);
            if (src.empty()) continue;

            Mat resized;
            resize(src, resized, fixedSize);
            gen_feature_input(resized, feature_file);
            gen_feature_output(label_file, 1, 2);
            success_count++;

            if (success_count % 100 == 0)
                cout << "Extracted " << success_count << " POSITIVE patterns...\n";
        }
    } else {
        cerr << "Warning: Positive directory not found -> " << images_dir_pos << "\n";
    }

    // ─────────── Negative (ไม่ใช่ทางม้าลาย: Label = 0) ───────────
    cout << "Reading NEGATIVE images from: " << images_dir_neg << "\n";
    if (fs::exists(images_dir_neg)) {
        int neg_count = 0;
        for (const auto& entry : fs::directory_iterator(images_dir_neg)) {
            string fp = entry.path().string();
            if (fp.find(".jpg") == string::npos &&
                fp.find(".png") == string::npos) continue;

            Mat src = imread(fp, IMREAD_COLOR);
            if (src.empty()) continue;

            Mat resized;
            resize(src, resized, fixedSize);
            gen_feature_input(resized, feature_file);
            gen_feature_output(label_file, 0, 2);
            success_count++;
            neg_count++;

            if (neg_count % 100 == 0)
                cout << "Extracted " << neg_count << " NEGATIVE patterns...\n";
        }
    } else {
        cerr << "Warning: Negative directory not found -> " << images_dir_neg << "\n";
    }

    cout << "\nDone! Total: " << success_count << " samples.\n";
    cout << "Saved to: " << feature_file << " and " << label_file << "\n";
    cout << "Each row = 160 features (120 col + 40 row)\n";
    return 0;
}

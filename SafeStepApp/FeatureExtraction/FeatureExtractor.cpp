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

// ฟังก์ชันสกัดคุณลักษณะ 1 มิติ (ตาม Lab: LprFeature)
void gen_feature_input(const Mat& image, const string& output_filename)
{
    float max_val = 0;
    vector<int> col_sums(image.cols, 0);

    Mat greyMat;
    if (image.channels() > 1) {
        cvtColor(image, greyMat, COLOR_BGR2GRAY);
    } else {
        greyMat = image.clone();
    }

    // คำนวณผลรวมความสว่างในแต่ละคอลัมน์
    for (int i = 0; i < greyMat.cols; i++)
    {
        int column_sum = 0;
        for (int k = 0; k < greyMat.rows; k++)
        {
            column_sum += greyMat.at<unsigned char>(k, i);
        }
        col_sums[i] = column_sum;
        if (col_sums[i] > max_val) {
            max_val = (float)col_sums[i];
        }
    }

    // เลี่ยงการหารด้วย 0
    if (max_val == 0) max_val = 1.0f;

    // บันทึกเป็น Normalize feature (0.0 ถึง 1.0) ลงไฟล์
    ofstream myfile;
    myfile.open(output_filename, ios::app);
    for (int i = 0; i < image.cols; i++)
    {
        float v = (float)col_sums[i] / max_val;
        myfile << v << ",";
    }
    myfile << endl;
    myfile.close();
}

// ฟังก์ชันระบุ Label (One-hot encoding)
// out: คลาสเป้าหมาย (0 = ไม่ใช่ทางม้าลาย, 1 = ทางม้าลาย)
// num: จำนวนคลาสทั้งหมด (ในกรณีนี้คือ 2)
void gen_feature_output(const string& output_filename, int out, int num = 2)
{
    ofstream myfile;
    myfile.open(output_filename, ios::app);
    for (int i = 0; i < num; i++)
    {
        if (i == out)
            myfile << 1 << ",";
        else
            myfile << 0 << ",";
    }
    myfile << endl;
    myfile.close();
}

int main(int argc, char** argv)
{
    // ---------------------------------------------------------------------------------
    // ใช้ Relative Path เพื่อให้รันที่เครื่องอื่นได้ (อิงจากตำแหน่งไฟล์ .vcxproj ของ Visual Studio)
    // หากรันแล้วไม่เจอโฟลเดอร์ ให้แก้ไข Working Directory ใน VS:
    // คลิกขวา Project -> Properties -> Debugging -> Working Directory -> ตั้งค่าเป็น $(ProjectDir)
    // ---------------------------------------------------------------------------------
    string images_dir_pos = "../images/"; 
    string images_dir_neg = "../images_non_zebra/"; 
    string feature_file = "features.txt";
    string label_file = "labels.txt";

    // ล้างไฟล์เก่าถ้ามี
    ofstream ofs(feature_file, ios::trunc); ofs.close();
    ofstream ols(label_file, ios::trunc); ols.close();

    cout << "--- SafeStep-AI Feature Extraction ---" << endl;

    int success_count = 0;
    Size fixedSize(120, 40);

    // -------------------------------------------------------------
    // ** 1. ประมวลผลข้อมูล Positive (ทางม้าลาย: Label = 1) **
    // -------------------------------------------------------------
    cout << "Reading POSITIVE images from: " << images_dir_pos << endl;
    if (fs::exists(images_dir_pos)) {
        for (const auto& entry : fs::directory_iterator(images_dir_pos))
        {
            string file_path = entry.path().string();
            if (file_path.find(".jpg") == string::npos && file_path.find(".png") == string::npos) continue; 

            Mat src = imread(file_path, IMREAD_COLOR);
            if (src.empty()) continue;

            Mat resized_src;
            resize(src, resized_src, fixedSize);
            
            gen_feature_input(resized_src, feature_file);
            gen_feature_output(label_file, 1, 2); // 1 = ทางม้าลาย
            success_count++;

            if (success_count % 100 == 0) cout << "Extracted " << success_count << " POSITIVE patterns..." << endl;
        }
    } else {
        cerr << "Warning: Positive Directory does not exist -> " << images_dir_pos << endl;
    }

    // -------------------------------------------------------------
    // ** 2. ประมวลผลข้อมูล Negative (ไม่ใช่ทางม้าลาย: Label = 0) **
    // -------------------------------------------------------------
    cout << "Reading NEGATIVE images from: " << images_dir_neg << endl;
    if (fs::exists(images_dir_neg)) {
        int neg_count = 0;
        for (const auto& entry : fs::directory_iterator(images_dir_neg))
        {
            string file_path = entry.path().string();
            if (file_path.find(".jpg") == string::npos && file_path.find(".png") == string::npos) continue; 

            Mat src = imread(file_path, IMREAD_COLOR);
            if (src.empty()) continue;

            Mat resized_src;
            resize(src, resized_src, fixedSize);
            
            gen_feature_input(resized_src, feature_file);
            gen_feature_output(label_file, 0, 2); // 0 = ไม่ใช่ทางม้าลาย
            success_count++;
            neg_count++;

            if (neg_count % 100 == 0) cout << "Extracted " << neg_count << " NEGATIVE patterns..." << endl;
        }
    } else {
        cerr << "Warning: Negative Directory does not exist -> " << images_dir_neg << endl;
        cerr << "Please create folder '" << images_dir_neg << "' and put non-zebra images inside!" << endl;
    }

    cout << "\nDone! Successfully extracted features from " << success_count << " total images." << endl;
    cout << "Outputs saved to: " << feature_file << " and " << label_file << endl;

    return 0;
}

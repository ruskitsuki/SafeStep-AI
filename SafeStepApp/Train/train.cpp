#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;

// ฟังก์ชันอ่านข้อมูลจากไฟล์ตัวหนังสือ (อิงตาม read_data_from_csv ใน LprLearn.cpp)
int read_data_from_csv(const string& filename, Mat& data, int n_samples, int n_samples_attributes)
{
    float tmpf;
    FILE* f = fopen(filename.c_str(), "r");
    if (!f)
    {
        printf("ERROR: cannot read file %s\n", filename.c_str());
        return 0;
    }

    for (int line = 0; line < n_samples; line++)
    {
        for (int attribute = 0; attribute < n_samples_attributes; attribute++)
        {
            if (fscanf(f, "%f,", &tmpf) != 1) {
                break;
            }
            data.at<float>(line, attribute) = tmpf;
        }
    }
    fclose(f);
    return 1; 
}

// ฟังก์ชันสำหรับ Train และ Save โมเดล
void train_ann_model(const string& data_in_filename, const string& data_out_filename, const string& model_save_filename)
{
    // กำหนด parameters (ควรสอดคล้องกับขนาด Feature สกัดที่ 120 คอลัมน์พิกเซล)
    // จำนวน Sample ผู้ใช้ต้องรู้ตอนทำ Feature Extraction หรือกำหนดแบบ Dynamic (ในตัวอย่างใช้วิธีนับไฟล์ไว้ก่อน)
    
    // ** คำเตือน: เราใช้ 120 Input Nodes (ความกว้างภาพ) และ 2 Output Nodes (One-hot 0/1) **
    int in_attributes = 120;
    int out_attributes = 2;

    // การนับหาค่าจำนวน Samples
    int samples = 0;
    string line;
    ifstream file(data_in_filename);
    while (getline(file, line)) {
        if (!line.empty()) samples++;
    }
    file.close();

    cout << "Total Samples found: " << samples << endl;
    if (samples == 0) {
        cerr << "Error: No data in " << data_in_filename << endl;
        return;
    }

    Mat data(samples, in_attributes, CV_32F);
    Mat responses(samples, out_attributes, CV_32F);

    cout << "Loading dataset..." << endl;
    if (!read_data_from_csv(data_in_filename, data, samples, in_attributes)) return;
    if (!read_data_from_csv(data_out_filename, responses, samples, out_attributes)) return;

    // 1. ตั้งค่าเลเยอร์ (Input: 120, Hidden: 15, Output: 2) เหมือน LprLearn.cpp ของอาจารย์
    int layer_sz[] = { in_attributes, 15, out_attributes };
    int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
    Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

    int method = ANN_MLP::BACKPROP; 
    double method_param = 0.001;
    int max_iter = 3000;

    Ptr<TrainData> tdata = TrainData::create(data, ROW_SAMPLE, responses);

    cout << "Training the ANN model (may take a few minutes)...\n";
    Ptr<ANN_MLP> model = ANN_MLP::create();
    
    model->setLayerSizes(layer_sizes);
    model->setActivationFunction(ANN_MLP::SIGMOID_SYM);
    model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, max_iter, 0.0001));
    model->setTrainMethod(method, method_param);

    model->train(tdata);
    cout << "Training completed!" << endl;

    // 2. บันทึกโมเดล
    model->save(model_save_filename);
    cout << "Model saved to " << model_save_filename << endl;
}

int main(int argc, char** argv)
{
    cout << "--- SafeStep-AI ANN Training ---" << endl;

    // ---------------------------------------------------------------------------------
    // ใช้ Relative Path อิงจากตำแหน่ง Working Directory ของ Visual Studio
    // ---------------------------------------------------------------------------------
    string feature_file = "features.txt";
    string label_file = "labels.txt";
    string model_file = "ZebraModel.xml";

    train_ann_model(feature_file, label_file, model_file);

    system("pause"); // หยุดหน้าจอรอดูผล
    return 0;
}

#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;

// ===========================================================================
// อ่านข้อมูล CSV ลงใน Mat
// ===========================================================================
int read_data_from_csv(const string& filename, Mat& data, int n_samples, int n_attributes)
{
    float tmpf;
    FILE* f = fopen(filename.c_str(), "r");
    if (!f) {
        printf("ERROR: cannot read file %s\n", filename.c_str());
        return 0;
    }
    for (int line = 0; line < n_samples; line++)
        for (int attr = 0; attr < n_attributes; attr++)
            if (fscanf(f, "%f,", &tmpf) == 1)
                data.at<float>(line, attr) = tmpf;
    fclose(f);
    return 1;
}

// ===========================================================================
// Train และ Save โมเดล ANN
// ===========================================================================
void train_ann_model(const string& data_in, const string& data_out, const string& model_path)
{
    // [FIX 6] in_attributes = 160 (120 col-sum + 40 row-sum)
    // ต้องตรงกับ FeatureExtractor.cpp และ ZebraDetector::ExtractFeature
    int in_attributes  = 160;
    int out_attributes = 2;

    // นับจำนวน samples จากไฟล์
    int samples = 0;
    {
        string line;
        ifstream file(data_in);
        while (getline(file, line))
            if (!line.empty()) samples++;
    }
    cout << "Total samples: " << samples << "\n";
    if (samples == 0) { cerr << "Error: No data in " << data_in << "\n"; return; }

    Mat data(samples, in_attributes, CV_32F);
    Mat responses(samples, out_attributes, CV_32F);

    cout << "Loading dataset...\n";
    if (!read_data_from_csv(data_in,  data,      samples, in_attributes))  return;
    if (!read_data_from_csv(data_out, responses, samples, out_attributes)) return;

    // โครงสร้าง: Input(160) → Hidden(20) → Output(2)
    // เพิ่ม Hidden nodes จาก 15 → 20 เนื่องจาก input มากขึ้น (160 vs 120)
    int layer_sz[] = { in_attributes, 20, out_attributes };
    int nlayers    = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
    Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

    int    method      = ANN_MLP::BACKPROP;
    double method_param = 0.001;   // learning rate
    int    max_iter    = 3000;

    Ptr<TrainData> tdata = TrainData::create(data, ROW_SAMPLE, responses);

    cout << "Training ANN (160→20→2), may take a few minutes...\n";
    Ptr<ANN_MLP> model = ANN_MLP::create();
    model->setLayerSizes(layer_sizes);
    model->setActivationFunction(ANN_MLP::SIGMOID_SYM);
    model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS,
                                        max_iter, 0.0001));
    model->setTrainMethod(method, method_param);
    model->train(tdata);

    cout << "Training completed!\n";
    model->save(model_path);
    cout << "Model saved to " << model_path << "\n";
}

// ===========================================================================
int main(int argc, char** argv)
{
    cout << "--- SafeStep-AI ANN Training ---\n";
    cout << "Using 160-feature model (120 col-sum + 40 row-sum)\n\n";

    string feature_file = "features.txt";
    string label_file   = "labels.txt";
    string model_file   = "ZebraModel.xml";

    train_ann_model(feature_file, label_file, model_file);

    system("pause");
    return 0;
}

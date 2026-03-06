#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class CrackInspector {
public:
    CrackInspector();
    bool loadModel(const std::string& modelPath);
    std::string openFileDialog();

    cv::Mat gen_feature_input(cv::Mat& image);
    float test_mlp_classifier(const std::string& filename_model, cv::Mat& dst);

private:
    cv::Ptr<cv::ml::ANN_MLP> model;
    const int INPUT_WIDTH = 1024;
    const int INPUT_HEIGHT = 1024;
};
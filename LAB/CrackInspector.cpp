
#include "pch.h"     
#define WIN32_LEAN_AND_MEAN 
#include <windows.h>  
#include <commdlg.h> 

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include "CrackInspector.h"

CrackInspector::CrackInspector() {}

// ‚À≈¥‚¡‡¥≈ ANN ∑’ËΩ÷°¡“·≈È«
bool CrackInspector::loadModel(const std::string& modelPath) {
    model = cv::ml::StatModel::load<cv::ml::ANN_MLP>(modelPath);
    return !model.empty();
}


std::string CrackInspector::openFileDialog() {
    OPENFILENAMEA ofn;
    char szFile[260];
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "JPEG Files\0*.jpg;*.jpeg\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn) == TRUE) return std::string(ofn.lpstrFile);
    return "";
}

cv::Mat CrackInspector::gen_feature_input(cv::Mat& image) {
    float maxVal = 0;
    std::vector<int> colSums(1024, 0);
    cv::Mat grey, resized;

    if (image.channels() == 3) cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    else grey = image.clone();

    cv::resize(grey, resized, cv::Size(1024, 1024));

    for (int i = 0; i < resized.cols; i++) {
        int sum = 0;
        for (int k = 0; k < resized.rows; k++) {
            sum += resized.at<unsigned char>(k, i);
        }
        colSums[i] = sum;
        if ((float)sum > maxVal) maxVal = (float)sum;
    }

    cv::Mat featureData(1, 1024, CV_32F);
    for (int i = 0; i < 1024; i++) {
        featureData.at<float>(0, i) = (maxVal > 0) ? (float)colSums[i] / maxVal : 0.0f;
    }
    return featureData;
}

// test_mlp_classifier
float CrackInspector::test_mlp_classifier(const std::string& filename_model, cv::Mat& dst) {
    if (model.empty()) {
        model = cv::ml::StatModel::load<cv::ml::ANN_MLP>(filename_model);
    }
    if (model.empty()) return -1.0f;

    return model->predict(dst);
}
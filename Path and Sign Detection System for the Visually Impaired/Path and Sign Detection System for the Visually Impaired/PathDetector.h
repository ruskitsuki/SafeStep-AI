#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class PathDetector
{
public:

    std::string detect(cv::Mat frame);

};
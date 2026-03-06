#include "pch.h"

#include <windows.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <io.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;


template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
	// load classifier from the specified file
	Ptr<T> model = StatModel::load<T>(filename_to_load);
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;

	return model;
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static bool
test_mlp_classifier(const string& filename_to_load,
	Mat data)
{
	Ptr<ANN_MLP> model;

	if (!filename_to_load.empty())
	{
		model = load_classifier<ANN_MLP>(filename_to_load);
		if (model.empty())
			return false;
	}

	float r = model->predict(data.row(0));
	printf("%f\n", r);

	return true;
}

Mat gen_feature_input(Mat& image)
{
	float max = 0;
	int val[120];

	Mat greyMat;
	cvtColor(image, greyMat, cv::COLOR_BGR2GRAY);

	for (int i = 0; i < greyMat.cols; i++)
	{
		int column_sum = 0;
		for (int k = 0; k < greyMat.rows; k++)
		{
			column_sum += greyMat.at<unsigned char>(k, i);
		}
		val[i] = column_sum;
		if (val[i] > max) max = (float)val[i];
	}

	Mat data(1, image.cols, CV_32F);

	for (int i = 0; i < image.cols; i++)
	{
		data.at<float>(0, i) = (float)val[i] / max;
	}
	return data;
}

int main(int argc, char** argv)
{
	string filename_image = "";
	string filename_model = "";
	int method = 0;
	int samples = 0;
	int in_attributes = 0;
	int out_attributes = 0;

	int i;
	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-load") == 0) // flag "-load filename.xml"
		{
			i++;
			filename_model = argv[i];
		}
		else if (strcmp(argv[i], "-image") == 0) // flag "-load filename.xml"
		{
			i++;
			filename_image = argv[i];
		}
	}

	Mat src, dst;
	src = imread(filename_image, 1);
	dst = gen_feature_input(src);

	test_mlp_classifier(filename_model, dst);

	return -1;
}
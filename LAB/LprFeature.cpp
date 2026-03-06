#include "pch.h"

#include <windows.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <io.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void gen_feature_input(Mat& image, string name)
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

	ofstream myfile;
	myfile.open(name, ios::app);

	for (int i = 0; i < image.cols; i++)
	{
		float v = (float)val[i] / max;
		myfile << v << ",";
	}
	myfile << endl;
	myfile.close();
}

void gen_feature_output(string name, int out, int num)
{
	ofstream myfile;
	myfile.open(name, ios::app);
	//myfile << out << ",";
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
	Mat src;

	ifstream myfile(argv[1]);

	int count;
	ifstream myfilein;
	myfilein.open("sample.txt", ios::in);
	myfilein >> count;
	myfilein.close();
	if (count < 0) count = 0;

	while (!myfile.eof())
	{
		string file;
		myfile >> file;

		if (file.empty()) break;

		src = imread(file, 1);

		if (!src.data)
		{
			cout << "Usage: ./LprFeature.exe <path_to_image> <textfile_name_input> <textfile_name_output> <output> <num_of_output>" << endl;
			break;
		}

		gen_feature_input(src, argv[2]);

		gen_feature_output(argv[3], atoi(argv[4]), atoi(argv[5]));

		count++;
	}

	ofstream myfileout;
	myfileout.open("sample.txt", ios::out);
	myfileout << count;
	myfileout.close();

	return 0;
}
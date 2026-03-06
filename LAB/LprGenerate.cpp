#include "pch.h"
#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <io.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void saveimage(Mat image, string path, int idx)
{
	path = path + to_string(idx) + ".jpg";
	cv::imwrite(path, image);
}

void translation(Mat& src, Mat& dst, int tx_l, int tx_r)
{
	Point2f srcTri[4];
	Point2f dstTri[4];

	Mat warp_mat(2, 3, CV_32FC1);

	/// Set the dst image the same type and size as src
	dst = Mat::zeros(src.rows, src.cols, src.type());

	/// Set your 3 points to calculate the  Affine Transform
	srcTri[0] = Point2f(0 + tx_l, 0);
	srcTri[1] = Point2f(src.cols - 1 + tx_r, 0);
	srcTri[2] = Point2f(0 + tx_l, src.rows - 1);
	srcTri[3] = Point2f(src.cols - 1 + tx_r, src.rows - 1);

	dstTri[0] = Point2f(0, 0);
	dstTri[1] = Point2f(src.cols - 1, 0);
	dstTri[2] = Point2f(0, src.rows - 1);
	dstTri[3] = Point2f(src.cols - 1, src.rows - 1);


	/// Get the Affine Transform
	warp_mat = getAffineTransform(srcTri, dstTri);

	/// Apply the Affine Transform just found to the src image
	warpAffine(src, dst, warp_mat, dst.size());
}

void rotation(Mat& src, Mat& dst, double angle)
{
	Mat rot_mat(2, 3, CV_32FC1);

	/// Compute a rotation matrix with respect to the center of the image
	Point center = Point(src.cols / 2, src.rows / 2);
	double scale = 1.0;

	/// Get the rotation matrix with the specifications above
	rot_mat = getRotationMatrix2D(center, angle, scale);

	/// Rotate the warped image
	warpAffine(src, dst, rot_mat, src.size(), 1, 0, Scalar(255, 255, 255, 1));
}

void blurr(Mat& image, string path, int& idx)
{
	int i, j;
	for (i = 3, j = 1; i <= 9; i = i + 4, j++)
	{
		GaussianBlur(image, image, Size(i, i), 0, 0);
		saveimage(image, path, idx);
		idx++;
	}
}

void noise(Mat& image, string path, int& idx)
{
	int i;

	for (i = 1; i < 4; i++)
	{
		cv::Mat noise = Mat(image.size(), CV_64F);
		Mat result;
		normalize(image, result, 0.0, 1.0, NORM_MINMAX, CV_64F);
		cv::randn(noise, 0, 0.1 + i / 20.0);
		result = result + noise;
		normalize(result, result, 0.0, 1.0, NORM_MINMAX, CV_64F);
		result.convertTo(result, CV_32F, 255, 0);

		// cv::imwrite(name + to_string(i) + "0.jpg" ,result);
		saveimage(image, path, idx);
		idx++;
		blurr(result, path, idx);
	}
}

int main(int argc, char** argv)
{
	Mat src;

	/// Load image
	src = imread(argv[1], 1);

	if (!src.data)
	{
		cout << "Usage: ./Generate.exe <path_to_image>" << endl;
		return -1;
	}

	std::stringstream path(argv[1]);
	std::string segment;
	std::vector<std::string> seglist;

	while (std::getline(path, segment, '.'))
	{
		seglist.push_back(segment);
	}

	cvtColor(src, src, COLOR_BGR2GRAY);

	string dir = seglist[0];

	system(("mkdir " + dir).c_str());

	dir = dir + "\\" + seglist[0];
	int idx = 0;

	Size size(120, 40);
	resize(src, src, size);
	Mat src_tx_left1;
	translation(src, src_tx_left1, 0, -10);
	Mat src_tx_left2;
	translation(src, src_tx_left2, 0, -20);
	Mat src_tx_right1;
	translation(src, src_tx_right1, 10, 0);
	Mat src_tx_right2;
	translation(src, src_tx_right2, 20, 0);
	Mat src_ro_left;
	rotation(src, src_ro_left, -5);
	Mat src_ro_right;
	rotation(src, src_ro_right, 5);

	// 1. resize
	saveimage(src, dir, idx);
	noise(src, dir, idx);

	// 2. left shift
	saveimage(src_tx_left1, dir, idx);
	noise(src_tx_left1, dir, idx);

	saveimage(src_tx_left2, dir, idx);
	noise(src_tx_left2, dir, idx);

	// 3. right shift
	saveimage(src_tx_right1, dir, idx);
	noise(src_tx_right1, dir, idx);

	saveimage(src_tx_right2, dir, idx);
	noise(src_tx_right2, dir, idx);

	// 4. rotate left
	saveimage(src_ro_left, dir, idx);
	noise(src_ro_left, dir, idx);

	saveimage(src_ro_right, dir, idx);
	noise(src_ro_right, dir, idx);

	cv::waitKey(0);

	return 0;
}
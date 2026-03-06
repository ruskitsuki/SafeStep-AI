#include "pch.h"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

RNG rng4(12345);

// Hu Moments
bool contour_features(vector<Point> ct)
{
    if (ct.size() < 5) return false;

    Moments mu = moments(ct);
    double hu[7];
    HuMoments(mu, hu);

    if (hu[0] < 0.18) return true;
    return false;
}

int main(int argc, char** argv)
{
    //img
    Mat frame, dst_image1, src_gray, grad, dst;

    if (argc < 2)
    {
        cout << "Error: Please provide an image path." << endl;
        cout << "Usage: " << argv[0] << " <path_to_image>" << endl;
        return -1;
    }

    string imagePath = argv[1];

    cout << "Loading image from: " << imagePath << endl;

    frame = imread(imagePath);

    if (frame.empty())
    {
        cout << "Could not open or find the image: " << imagePath << std::endl;
        return -1;
    }

    //img processing
    GaussianBlur(frame, dst_image1, Size(5, 5), 0, 0);
    cvtColor(dst_image1, src_gray, COLOR_BGR2GRAY);

    //sobel
    Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
    Sobel(src_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    Sobel(src_gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    threshold(grad, dst, 30, 255, THRESH_BINARY_INV); // ใช้ Binary Inverse ตาม Logic ของคุณ

    //contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dst, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    double max_area = 0.0;
    Rect max_rect;
    bool found_any = false;

    for (size_t i = 0; i < contours.size(); i++)
    {
        vector<Point> poly;
        approxPolyDP(Mat(contours[i]), poly, 1, true);

        if (poly.size() > 5 && contour_features(poly))
        {
            double current_area = contourArea(poly);
            if (current_area > max_area)
            {
                max_area = current_area;
                max_rect = boundingRect(poly);
                found_any = true;
            }
        }
    }

    if (!found_any) {
        cout << "No suitable contours found." << endl;
        return -1;
    }

    Mat croppedImage = src_gray(max_rect);
    int rotation_count = 0;

    while (rotation_count < 4) {
        Mat imgPolar;
        linearPolar(croppedImage, imgPolar, Point2f(croppedImage.cols / 2, croppedImage.rows / 2),
            croppedImage.cols / 2, INTER_LINEAR + WARP_FILL_OUTLIERS);

        Mat imgEdge;
        Mat croppedEdgeRef(imgPolar, Rect(imgPolar.cols - 50, 0, 50, imgPolar.rows));
        croppedEdgeRef.copyTo(imgEdge);

        rotate(imgEdge, imgEdge, ROTATE_90_COUNTERCLOCKWISE);

        int maxhist = 0;
        int hist[1000] = { 0 };

        for (int x = 0; x < imgEdge.cols; x++) {
            int sum = 0;
            for (int y = 0; y < imgEdge.rows; y++) {
                sum += imgEdge.at<unsigned char>(y, x);
            }
            hist[x] = sum;
            if (maxhist < sum) maxhist = sum;
        }

        int histnor[1000] = { 0 };
        int histth[1000] = { 0 };
        for (int x = 0; x < imgEdge.cols; x++) {
            if (maxhist > 0) histnor[x] = (float)hist[x] / maxhist * 100;

            //ค่า Threshold
            if (hist[x] > 5000) histth[x] = 255;
            else histth[x] = 0;
        }

        //หาขอบเขต
        int xleft = 0;
        for (int x = 0; x < imgEdge.cols; x++) {
            if (histth[x] == 0) { xleft = x; break; }
        }

        int xright = imgEdge.cols - 1;
        for (int x = xright; x >= 0; x--) {
            if (histth[x] == 0) { xright = x; break; }
        }

        printf("Iteration %d: Left %d, Right %d\n", rotation_count, xleft, xright);

        if (xleft == 0 || xright == imgEdge.cols - 1) {
            rotate(croppedImage, croppedImage, ROTATE_90_CLOCKWISE);
            rotation_count++;
            continue;
        }
        else {
            Mat imgHist = Mat::zeros(Size(imgEdge.cols, 100), CV_8UC3);
            for (int x = 0; x < imgEdge.cols; x++) {
                line(imgHist, Point(x, 100), Point(x, 100 - histnor[x]), Scalar(255, 255, 255), 1);
                if (x == xleft || x == xright)
                    line(imgHist, Point(x, 0), Point(x, 100), Scalar(0, 0, 255), 1); // ขอบเขตที่หาได้
            }

            imshow("1. Image", croppedImage);
            imshow("2. Polar Transform", imgPolar);
            imshow("3. Histogram", imgHist);

            waitKey(0);
            break;
        }
    }

    return 0;
}
// lab8.cpp : This file contains the 'main' function. Program execution begins and ends there.

//



#include "pch.h"

#include <iostream>



#include "opencv2/opencv.hpp"



#include "opencv2/objdetect.hpp"

#include "opencv2/highgui.hpp"

#include "opencv2/imgproc.hpp"



#include <stdio.h>



using namespace std;

using namespace cv;



/** Function Headers */

void detectAndDisplay1(Mat frame);



/** Global variables */



String face_cascade_name1 = "cascade.xml";

CascadeClassifier face_cascade1;

String window_name1 = "Object Detection Result";



/** @function main */


int main(void)
{
    // Load the cascades
    if (!face_cascade1.load(face_cascade_name1)) {
        printf("--(!)Error loading cascade.xml\n");
        return -1;
    };

    FILE* fp;
    errno_t err = fopen_s(&fp, "test.dat", "r");
    if (err != 0 || fp == NULL) {
        printf("Error opening test.dat\n");
        return -1;
    }

    Mat frame;
    char name[100];

    while (fscanf_s(fp, "%s", name, (unsigned)_countof(name)) != EOF)
    {
        printf("Reading image: %s\n", name);
        frame = imread(name);

        if (frame.empty()) {
            printf(" --(!) Image not found: %s\n", name);
            break;
        }

        //วาดสี่เหลี่ยม
        detectAndDisplay1(frame);

        // บันทึกไฟล์
        string outputName = "result_" + string(name); 

        bool isSuccess = imwrite(outputName, frame);

        if (isSuccess) {
            printf("Successfully saved: %s\n", outputName.c_str());
        }
        else {
            printf("---(!) Failed to save: %s\n", outputName.c_str());
        }
 

        int c = waitKey(0);
        if ((char)c == 27) { break; }
    }

    fclose(fp);
    return 0;
}



/** @function detectAndDisplay */

void detectAndDisplay1(Mat frame)
{
	std::vector<Rect> objects;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(1.0); // ปรับค่าความคมชัด
	clahe->apply(frame_gray, frame_gray);
	//equalizeHist(frame_gray, frame_gray);
	GaussianBlur(frame_gray, frame_gray, Size(5, 5), 0);

	// ปรับพารามิเตอร์ตามงาน
	face_cascade1.detectMultiScale(frame_gray, objects, 1.05, 8, 0 | CASCADE_SCALE_IMAGE, Size(95, 95));

	for (const auto& obj : objects)
	{
		rectangle(frame, obj, Scalar(0, 255, 0), 2);
	}

	// แสดงผล
	double scale = 0.5;
	Mat resizedFrame;
	resize(frame, resizedFrame, Size(), scale, scale);
	imshow(window_name1, resizedFrame);
}
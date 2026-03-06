#include <opencv2/opencv.hpp>
#include <iostream>
#include "ZebraDetector.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    cout << "--- SafeStep-AI Live Demo ---" << endl;

    ZebraDetector detector;

    // ---------------------------------------------------------------------------------
    // ใช้ Relative Path อิงจากตำแหน่ง Working Directory ของ Visual Studio
    // ---------------------------------------------------------------------------------
    string modelPath = "ZebraModel.xml"; 
    
    if (!detector.LoadModel(modelPath))
    {
        cerr << "Failed to start. Model not found!" << endl;
        system("pause");
        return -1;
    }

    // ---------------------------------------------------------------------------------
    // ให้ผู้ใช้สามารถใส่ Path ของวิดีโอผ่าน Command Line Argument ได้
    // หรือถ้าไม่ใส่โปรแกรมจะถามให้พิมพ์ Path วิดีโอเอง
    // ---------------------------------------------------------------------------------
    string videoSource;
    if (argc > 1) {
        videoSource = argv[1];
    } else {
        cout << "Enter video path (or press '0' for Webcam): ";
        getline(cin, videoSource);
    }
    
    // ลบเครื่องหมาย " (Double Quote) ที่ติดมาตอน Copy as path ใน Windows
    if (!videoSource.empty() && videoSource.front() == '"' && videoSource.back() == '"') {
        videoSource = videoSource.substr(1, videoSource.length() - 2);
    }
    
    VideoCapture cap;
    
    // ลองเปิดเป็นหมายเลข (Webcam) ก่อน ถ้าไม่เจอ/ไม่ได้พิมพ์ตัวเลข ก็เปิดเป็นไฟล์ Path
    if (videoSource.length() == 1 && isdigit(videoSource[0])) {
        cap.open(stoi(videoSource));
    } else {
        // ใช้ไฟล์ Path เช่น "video.mp4" หรือ "C:/user/video.mp4"
        cap.open(videoSource);
    }

    if (!cap.isOpened())
    {
        cerr << "Error: Could not open video stream." << endl;
        system("pause");
        return -1;
    }

    cout << "Press 'ESC' to exit." << endl;

    Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "End of video stream!" << endl;
            break;
        }

        // นำเข้าขั้นตอนการดึงรูป สแกน วาดกรอบ
        detector.DetectAndDraw(frame);

        imshow("SafeStep-AI Demo", frame);

        char c = (char)waitKey(30); 
        if (c == 27) // เลิกเล่นเมื่อกดปุ่ม ESC
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}

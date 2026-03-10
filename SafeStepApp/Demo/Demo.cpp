#include <opencv2/opencv.hpp>
#include <iostream>
#include "ZebraDetector.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    cout << "--- SafeStep-AI Live Demo ---\n";
    cout << "Keys: [ESC] Exit  |  [D] Toggle Debug Windows\n\n";

    ZebraDetector detector;

    string modelPath = "ZebraModel.xml";
    if (!detector.LoadModel(modelPath))
    {
        cerr << "Failed to start. Model not found!\n";
        system("pause");
        return -1;
    }

    // รับ Path วิดีโอจาก Command Line หรือถามผู้ใช้
    string videoSource;
    if (argc > 1) {
        videoSource = argv[1];
    } else {
        cout << "Enter video path (or '0' for Webcam): ";
        getline(cin, videoSource);
    }

    // ลบ Double Quote ที่ติดมาจาก Copy as path
    if (!videoSource.empty() && videoSource.front() == '"' && videoSource.back() == '"')
        videoSource = videoSource.substr(1, videoSource.length() - 2);

    VideoCapture cap;
    if (videoSource.length() == 1 && isdigit(videoSource[0]))
        cap.open(stoi(videoSource));
    else
        cap.open(videoSource);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open video stream.\n";
        system("pause");
        return -1;
    }

    // อ่าน frame แรกเพื่อเช็คว่าวิดีโอเปิดติดไหม และใช้คำนวณสัดส่วนหน้าต่างโปรแกรม
    Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) {
        cerr << "Error: Empty video source.\n";
        return -1;
    }

    cout << "\nFrame size: " << firstFrame.cols << " x " << firstFrame.rows << "\n";
    cout << "\nPress [ESC] to exit, [D] to toggle Debug Windows.\n";

    // ─────────── Setup Main Window ───────────
    namedWindow("SafeStep-AI Demo", WINDOW_NORMAL);
    int mainH = 600;
    int mainW = (int)((float)firstFrame.cols / firstFrame.rows * mainH);
    resizeWindow("SafeStep-AI Demo", mainW, mainH);

    // เช็ค FPS ของวิดีโอเพื่อใช้คำนวณการเล่นความเร็วปกติ
    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0 || fps > 120) fps = 30.0; // ค่าเริ่มต้นถ้าอ่านไม่ติด หรือได้ค่าแปลกๆ
    int64 startTime = getTickCount();
    long long frameCount = 1;

    // ─────────── Main Loop ───────────
    Mat frame = firstFrame;
    while (true)
    {
        if (frame.empty()) {
            cout << "End of video stream!\n";
            break;
        }

        // ย่อภาพเพื่อไม่ให้กิน CPU เยอะเกินไป
        if (frame.cols > 1280) {
            float scale = 1280.0f / frame.cols;
            resize(frame, frame, Size(), scale, scale);
        }

        detector.DetectAndDraw(frame);

        // Overlay: แสดง status ของ Debug
        string dbgStatus = detector.IsDebugMode()
                           ? "[D] Debug: ON"
                           : "[D] Debug: OFF";
        putText(frame, dbgStatus, Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 200, 255), 2);

        imshow("SafeStep-AI Demo", frame);

        // -- คำนวณเวลาเพื่อควบคุมความเร็วให้พอดีกับความจริง --
        double elapsedSec = (getTickCount() - startTime) / getTickFrequency();
        long long expectedFrame = (long long)(elapsedSec * fps);

        int delayMs = 1;
        if (frameCount > expectedFrame) {
            // ถ้าประมวลผลเร็วกว่าเวลาวิดีโอ ให้รอ (หน่วงเวลา) เพื่อให้ความเร็วพอดี
            double targetTimeSec = (double)frameCount / fps;
            delayMs = (int)((targetTimeSec - elapsedSec) * 1000.0);
            if (delayMs < 1) delayMs = 1;
        }

        char c = (char)waitKey(delayMs);
        if (c == 27) break;  // ESC

        // [DEBUG] กด D เพื่อ toggle หน้าต่าง debug (intermediate steps)
        if (c == 'd' || c == 'D') {
            bool nextDebug = !detector.IsDebugMode();
            detector.SetDebugMode(nextDebug);
            if (!nextDebug) {
                // ปิดหน้าต่าง debug ทั้งหมดอย่างปลอดภัย
                try { cv::destroyWindow("Debug 1: L-Channel + CLAHE"); } catch (...) {}
                try { cv::destroyWindow("Debug 2: ROI Trapezoid Mask"); } catch (...) {}
                try { cv::destroyWindow("Debug 3: Threshold"); } catch (...) {}
                try { cv::destroyWindow("Debug 4: Morph Close & Fill"); } catch (...) {}
                try { cv::destroyWindow("Debug 5: Valid Stripes"); } catch (...) {}
            }
        }

        // ถ้าประมวลผลช้ากว่าวิดีโอ (สโลวโมชัน) ให้รันผ่าน (Skip) เฟรมที่ไม่ทันทิ้งไปเพื่อตามเวลาจริง
        elapsedSec = (getTickCount() - startTime) / getTickFrequency();
        expectedFrame = (long long)(elapsedSec * fps);
        while (frameCount < expectedFrame) {
            if (!cap.grab()) break; // ถ้า Grab ไม่ได้แปลว่าสุดวิดีโอแล้ว
            frameCount++;
        }

        // อ่าน frame ถัดไปที่จะนำมาแสดงและประมวลผล
        cap >> frame;
        frameCount++;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

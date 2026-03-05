#include "pch.h"
// ============================================================
//  main_demo.cpp
//  โปรแกรมทดสอบ ZebraDetector แบบ Real-time (กล้อง หรือ วิดีโอ)
//
//  วิธีใช้: รัน exe นี้หลังจากมี ZebraModel.xml แล้ว
//           กด 'q' หรือ ESC เพื่อออก
// ============================================================
#include <opencv2/opencv.hpp>
#include "ZebraDetector.h"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    cout << "=====================================" << endl;
    cout << "   Zebra Crossing Detection Demo     " << endl;
    cout << "=====================================" << endl;

    // ---- โหลด Detector ----
    ZebraDetector detector("ZebraModel.xml");
    if (!detector.isReady()) {
        cerr << "[ERROR] โหลดโมเดลไม่สำเร็จ กรุณา Train โมเดลก่อนด้วย train.cpp" << endl;
        return -1;
    }

    // ---- เปิดกล้องหรือวิดีโอ ----
    VideoCapture cap;
    if (argc > 1) {
        // รับ path วิดีโอจาก command line argument
        cap.open(argv[1]);
        cout << "[INFO] เปิดวิดีโอ: " << argv[1] << endl;
    } else {
        // ใช้กล้อง webcam (index 0)
        cap.open(0);
        cout << "[INFO] เปิดกล้อง webcam (index 0)" << endl;
    }

    if (!cap.isOpened()) {
        cerr << "[ERROR] ไม่สามารถเปิดกล้อง/วิดีโอได้" << endl;
        return -1;
    }

    // ---- Main Loop ----
    Mat frame;
    int frameCount = 0;
    double totalTime = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "[INFO] สิ้นสุดวิดีโอ" << endl;
            break;
        }

        // วัดเวลา processing
        double t1 = (double)getTickCount();
        bool found = detector.DetectAndDraw(frame);
        double t2 = (double)getTickCount();
        double elapsed = (t2 - t1) / getTickFrequency() * 1000.0; // ms

        frameCount++;
        totalTime += elapsed;

        // แสดง FPS และสถานะ
        double fps = 1000.0 / (totalTime / frameCount);
        string statusText = found ? "ZEBRA DETECTED!" : "No zebra";
        Scalar statusColor = found ? Scalar(0, 255, 0) : Scalar(0, 165, 255);

        // overlay ข้อมูล
        putText(frame,
                "FPS: " + to_string((int)fps),
                Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
        putText(frame,
                statusText,
                Point(10, 65),
                FONT_HERSHEY_SIMPLEX, 0.8, statusColor, 2);
        putText(frame,
                "Press 'q' to quit",
                Point(10, frame.rows - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

        imshow("Zebra Crossing Detector", frame);

        // กด q หรือ ESC เพื่อออก
        int key = waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    cout << "[INFO] เฉลี่ย processing time: "
         << totalTime / frameCount << " ms/frame" << endl;
    cout << "[DONE] ปิดโปรแกรม" << endl;

    cap.release();
    destroyAllWindows();
    return 0;
}

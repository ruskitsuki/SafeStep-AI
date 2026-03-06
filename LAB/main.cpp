#include "pch.h"
#include "CrackInspector.h"
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    CrackInspector inspector;
    string filename_model = "model.xml"; 

    if (!inspector.loadModel(filename_model)) {
        cout << "Error: " << filename_model << " not found!" << endl;
        return -1;
    }

    namedWindow("Crack Detection System", WINDOW_NORMAL);
    resizeWindow("Crack Detection System", 1024, 768);

    bool running = true;
    while (running) {
        Mat welcomeImg = Mat::zeros(600, 800, CV_8UC3);
        putText(welcomeImg, "Press 'O' to Open Image", Point(150, 250), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 2);
        putText(welcomeImg, "Press 'ESC' to Exit", Point(250, 320), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200, 200, 200), 1);
        imshow("Crack Detection System", welcomeImg);

        int key = waitKey(0);
        int cleanKey = key & 0xFF;

        if (cleanKey == 'o' || cleanKey == 'O' || cleanKey == 185 || cleanKey == 207) {
            string path = inspector.openFileDialog();
            if (!path.empty()) {
                Mat src = imread(path);
                if (src.empty()) continue;

              
                Mat imgRoi = src;
                Mat dst = inspector.gen_feature_input(imgRoi); // dst = gen_feature_input(imgRoi)
                float result = inspector.test_mlp_classifier(filename_model, dst); // test_mlp_classifier

                // Result in cmd
                cout << "Path: " << path << endl;
                if (result == 0) cout << "Result: CRACK" << endl;
                else cout << "Result: NO-CRACK" << endl;
                cout << "----------------------------" << endl;

                // Result
                Scalar color = (result == 0) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
                string label = (result == 0) ? "CRACK" : "NO-CRACK";

                rectangle(src, Rect(0, 0, src.cols, src.rows), color, 15);
                putText(src, label, Point(50, 80), FONT_HERSHEY_SIMPLEX, 2.5, color, 8);
                putText(src, "Press ANY KEY to go BACK", Point(50, src.rows - 50), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 3);

                imshow("Crack Detection System", src);
                waitKey(0);
            }
        }
        else if (cleanKey == 27) running = false;
    }

    destroyAllWindows();
    return 0;
}
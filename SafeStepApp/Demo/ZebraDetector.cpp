#include "ZebraDetector.h"
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::ml;

ZebraDetector::ZebraDetector()
{
}

ZebraDetector::~ZebraDetector()
{
}

bool ZebraDetector::LoadModel(const string& modelPath)
{
    model = StatModel::load<ANN_MLP>(modelPath);
    if (model.empty()) {
        cout << "Error: Could not read the classifier " << modelPath << endl;
        return false;
    }
    cout << "The classifier " << modelPath << " is loaded successfully.\n";
    return true;
}

Mat ZebraDetector::ExtractFeature(Mat& cropImage)
{
    // ย่อขนาดให้ตรงกับตอน Train คือ 120x40
    Mat resizedImg;
    resize(cropImage, resizedImg, Size(120, 40));

    float max_val = 0;
    vector<int> col_sums(resizedImg.cols, 0);

    Mat greyMat;
    if (resizedImg.channels() > 1) {
        cvtColor(resizedImg, greyMat, COLOR_BGR2GRAY);
    } else {
        greyMat = resizedImg.clone();
    }

    // คำนวณ Column sum (เหมือนใน FeatureExtractor.cpp)
    for (int i = 0; i < greyMat.cols; i++)
    {
        int column_sum = 0;
        for (int k = 0; k < greyMat.rows; k++)
        {
            column_sum += greyMat.at<unsigned char>(k, i);
        }
        col_sums[i] = column_sum;
        if (col_sums[i] > max_val) max_val = (float)col_sums[i];
    }

    if (max_val == 0) max_val = 1.0f;

    // เตรียม Mat สำหรับส่งให้ Model Predict (1 แถว, 120 คอลัมน์)
    Mat data(1, resizedImg.cols, CV_32F);
    for (int i = 0; i < resizedImg.cols; i++)
    {
        data.at<float>(0, i) = (float)col_sums[i] / max_val;
    }
    return data;
}

void ZebraDetector::DetectAndDraw(Mat& frame)
{
    if (frame.empty() || model.empty()) return;

    Mat gray, thresh, morph;
    
    // 1. แปลงเป็นภาพขาวดำ
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // 2. Thresholding ดึงเฉพาะสีขาวสว่างๆ ออกมา (ปรับค่าเอาตามแสงวิดีโอ)
    threshold(gray, thresh, 180, 255, THRESH_BINARY);

    // 3. Morphological Operations เพื่อเชื่อมโยงทางม้าลายเป็นก้อน (Blob)
    // ** เปลี่ยน Kernel เป็นขนาดใหญ่ (เช่น 51x51) เพื่อให้แถบม้าลายหลอมรวมเป็นก้อนหน้ากระดานแผ่นเดียว **
    Mat kernel = getStructuringElement(MORPH_RECT, Size(61, 61)); 
    morphologyEx(thresh, morph, MORPH_CLOSE, kernel);

    // เอาไว้ดูภาพ debug (ถ้าหาทางม้าลายไม่เจอ สามารถปลดคอมเมนต์เพื่อดูว่าก้อนขาวมันเชื่อมกันมั้ย)
    // imshow("Zebra Morph Debug", morph);

    // 4. หาขอบ Contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(morph, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        // กรองขนาดพื้นที่ (Contour Area) ที่ต้องใหญ่พอสมควรสำหรับทางม้าลายในมุมกล้อง
        double area = contourArea(contours[i]);
        if (area < 8000) continue; // ทางม้าลายจริงจะเต็มจอและมี Area ใหญ่มาก (ปรับขึ้นเป็น 8000+)

        Rect bounding_rect = boundingRect(contours[i]);

        // ** แก้ไข Logic คัดกรองใหม่ (รองรับกล่องขนาดใหญ่ที่เกิดจากการรวบเส้น) **
        // 1. ความสูงของ Box ต้องมากพอ: กล่องทางม้าลายรวมก้อนแล้วควรจะหนาพอสมควร
        if (bounding_rect.height < 50) continue;

        // 2. สัดส่วนภาพ (Aspect Ratio):
        // กล่องทางม้าลายรวมก้อนมักจะแบนแนวนอน (ครอบทั้งถนน) แต่อาจจะมีความหนามาก
        float aspectRatio = (float)bounding_rect.width / (float)bounding_rect.height;
        if (aspectRatio < 0.5 || aspectRatio > 10.0) continue; // เปิดกว้างขึ้นไม่ตัดทิ้งง่ายๆ

        // 3. ตำแหน่งจุดศูนย์กลาง: ควรอยู่ค่อนไปทางพื้นถนน
        int center_y = bounding_rect.y + (bounding_rect.height / 2);
        if (center_y < frame.rows / 3) continue; // หย่อนให้ขึ้นไปได้ถึง 1 ใน 3 ของจอบน เผื่อกล่องใหญ่มาก

        // กัน BoundingBox เกินขอบเขตภาพ
        if (bounding_rect.x < 0) bounding_rect.x = 0;
        if (bounding_rect.y < 0) bounding_rect.y = 0;
        if (bounding_rect.x + bounding_rect.width >= frame.cols) bounding_rect.width = frame.cols - bounding_rect.x - 1;
        if (bounding_rect.y + bounding_rect.height >= frame.rows) bounding_rect.height = frame.rows - bounding_rect.y - 1;

        // 5. ตัดรูปเตรียมตรวจสอบ Model (Crop)
        Mat cropImg = frame(bounding_rect);
        
        // 6. สกัดคุณลักษณะแล้วดึงไปใส่โมเดล
        Mat featureData = ExtractFeature(cropImg);
        
        Mat predict_responses;
        model->predict(featureData, predict_responses); // ทายผล

        // เช็คหาว่าความน่าจะเป็นตกที่ Class 0 หรือ Class 1 มากกว่า
        float max_prob = predict_responses.at<float>(0, 0);
        int max_class_id = 0;
        for (int c = 1; c < predict_responses.cols; c++)
        {
            if (predict_responses.at<float>(0, c) > max_prob)
            {
                max_prob = predict_responses.at<float>(0, c);
                max_class_id = c;
            }
        }

        // สมมติคลาสรับ 1 คือทางม้าลาย (จากที่เรากำหนดตอนเขียน Train)
        if (max_class_id == 1)
        {
            // วาดกรอบสีเขียวล้อมรอบ
            rectangle(frame, bounding_rect, Scalar(0, 255, 0), 3);
            putText(frame, "Zebra Crossing", Point(bounding_rect.x, bounding_rect.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        }
    }
}

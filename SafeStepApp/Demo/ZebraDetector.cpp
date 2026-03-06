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

    Mat gray, thresh, edge;

    // 1. แปลงเป็นภาพขาวดำ
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // 2. Thresholding ดึงเฉพาะสีขาว (ปรับแสง)
    threshold(gray, thresh, 180, 255, THRESH_BINARY);

    // 3. ใช้ Morphological ขนาดเล็กแค่ลบ Noise รอยแตกของซี่ (ไม่ได้เชื่อมก้อนแล้ว)
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);

    // 4. หาขอบ Contours (หาซี่เดี่ยวๆ)
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> validStripes;
    double screen_area = frame.rows * frame.cols;

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        Rect rect = boundingRect(contours[i]);

        // ** กรองหาเฉพาะสิ่งที่ "หน้าตาเหมือน 1 ซี่ทางม้าลาย" แบนๆ ยาวๆ หรือเฉียงๆ **
        if (area > 300 && area < screen_area * 0.15) // ขนาดซี่กำลังดี ไม่ใช่เส้นด้าย และไม่ใช่กำแพงยักษ์
        {
            float aspectRatio = (float)rect.width / (float)rect.height;

            // ซี่ทางม้าลายส่วนใหญ่จะกว้างกว่าความสูง
            if (aspectRatio > 0.8 && aspectRatio < 8.0) 
            {
                // ต้องไม่ได้ลอยอยู่บนฟ้า
                if (rect.y > frame.rows / 4) 
                {
                    validStripes.push_back(rect);
                    // วาดกรอบสีน้ำเงินเล็กๆ ดูซี่ (ลบคอมเมนต์ได้ถ้าอยากดู debug)
                    // rectangle(frame, rect, Scalar(255, 0, 0), 2);
                }
            }
        }
    }

    // 5. นำซี่ทั้งหมดมาหลอมรวมกัน (Merge Stripes)
    // ก่อนรวม: ต้องผ่านด่านตรวจสอบว่าซี่เหล่านี้หน้าตาเหมือนทางม้าลายจริงๆ

    // ด่าน 1: ต้องมีซี่อย่างน้อย 3 ซี่ขึ้นไป
    if (validStripes.size() >= 3)
    {
        // ด่าน 2: แต่ละซี่ต้องกว้างพอ (อย่างน้อย 30% ของจอ = คาดว่าพาดเต็มถนน)
        // กรองซี่แคบๆ อย่างตัวอักษรรายตัวออกก่อน
        vector<Rect> wideStripes;
        for (const auto& s : validStripes) {
            if (s.width >= frame.cols * 0.30) {
                wideStripes.push_back(s);
            }
        }
        if (wideStripes.size() < 3) return; // ไม่พอ

        // ด่าน 3: ซี่ต้องมีความกว้างใกล้เคียงกัน (สม่ำเสมอคล้ายลาย)
        float totalWidth = 0;
        for (const auto& s : wideStripes) totalWidth += s.width;
        float avgWidth = totalWidth / wideStripes.size();
        int passSimilar = 0;
        for (const auto& s : wideStripes) {
            if (abs(s.width - avgWidth) < avgWidth * 0.5f) passSimilar++;
        }
        if (passSimilar < (int)wideStripes.size() * 0.7) return; // ส่วนใหญ่ต้องใกล้เคียงกัน

        // ด่าน 4: ซี่ต้องกระจายตัวในแกน Y (เรียงซ้อนกัน ไม่ใช่วางเรียงข้างๆ กัน)
        // หาค่า Y min/max ดูว่าช่วง Y ที่กลุ่มซี่กินนั้นกว้างกว่าซี่เดี่ยวๆ กี่เท่า
        int y_min = frame.rows, y_max = 0;
        float avgHeight = 0;
        for (const auto& s : wideStripes) {
            if (s.y < y_min) y_min = s.y;
            if (s.y + s.height > y_max) y_max = s.y + s.height;
            avgHeight += s.height;
        }
        avgHeight /= wideStripes.size();
        int y_span = y_max - y_min;
        // ช่วง Y ต้องกินพื้นที่มากกว่า 2.5 เท่าของซี่เดี่ยว (= มีซี่หลายชั้นซ้อนกัน)
        if (y_span < avgHeight * 2.5f) return;

        // ผ่านทุกด่านแล้ว! รวบซี่เป็นกล่องเดียว
        int min_x = frame.cols, min_y = frame.rows;
        int max_x = 0, max_y = 0;
        for (const auto& stripe : wideStripes)
        {
            if (stripe.x < min_x) min_x = stripe.x;
            if (stripe.y < min_y) min_y = stripe.y;
            if (stripe.x + stripe.width > max_x) max_x = stripe.x + stripe.width;
            if (stripe.y + stripe.height > max_y) max_y = stripe.y + stripe.height;
        }

        Rect bounding_rect(min_x, min_y, max_x - min_x, max_y - min_y);

        int padding = 15;
        bounding_rect.x = max(0, bounding_rect.x - padding);
        bounding_rect.y = max(0, bounding_rect.y - padding);
        bounding_rect.width = min(frame.cols - bounding_rect.x, bounding_rect.width + 2 * padding);
        bounding_rect.height = min(frame.rows - bounding_rect.y, bounding_rect.height + 2 * padding);

        if (bounding_rect.width > 0 && bounding_rect.height > 0)
        {
            // 6. ส่งรูปในกรอบรวมให้ AI ทายผลครั้งสุดท้าย
            Mat cropImg = frame(bounding_rect);
            Mat featureData = ExtractFeature(cropImg);

            Mat predict_responses;
            model->predict(featureData, predict_responses);

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

            if (max_class_id == 1) // 1 คือคลาสทางม้าลาย
            {
                rectangle(frame, bounding_rect, Scalar(0, 255, 0), 4);
                putText(frame, "Zebra Crossing", Point(bounding_rect.x, bounding_rect.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 3);
            }
        }
    }
}

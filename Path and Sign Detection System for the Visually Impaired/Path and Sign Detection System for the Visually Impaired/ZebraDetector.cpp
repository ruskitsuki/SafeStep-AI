#include "ZebraDetector.h"
#include <vector>
#include <algorithm>
#include <climits>

using namespace cv;
using namespace std;
using namespace cv::ml;

ZebraDetector::ZebraDetector() {}
ZebraDetector::~ZebraDetector() {}

void ZebraDetector::SetDebugMode(bool enabled) {
    debugMode = enabled;
    if (debugMode) {
        // à¸•à¸±à¹‰à¸‡à¸„à¹ˆà¸²à¸«à¸™à¹‰à¸²à¸•à¹ˆà¸²à¸‡ Debug à¹ƒà¸«à¹‰à¸¢à¹ˆà¸­-à¸‚à¸¢à¸²à¸¢à¹„à¸”à¹‰ à¹à¸¥à¸°à¸à¸³à¸«à¸™à¸”à¸‚à¸™à¸²à¸”à¹€à¸£à¸´à¹ˆà¸¡à¸•à¹‰à¸™à¹ƒà¸«à¹‰à¹€à¸¥à¹‡à¸à¸žà¸­à¸¡à¸­à¸‡à¹€à¸«à¹‡à¸™à¸žà¸£à¹‰à¸­à¸¡à¸à¸±à¸™
        int winW = 350;
        int winH = 350;

        namedWindow("Debug 1: L-Channel + CLAHE", WINDOW_NORMAL);
        resizeWindow("Debug 1: L-Channel + CLAHE", winW, winH);

        namedWindow("Debug 2: ROI Trapezoid Mask", WINDOW_NORMAL);
        resizeWindow("Debug 2: ROI Trapezoid Mask", winW, winH);

        namedWindow("Debug 3: Threshold", WINDOW_NORMAL);
        resizeWindow("Debug 3: Threshold", winW, winH);

        namedWindow("Debug 4: Morph Close & Fill", WINDOW_NORMAL);
        resizeWindow("Debug 4: Morph Close & Fill", winW, winH);

        namedWindow("Debug 5: Valid Stripes", WINDOW_NORMAL);
        resizeWindow("Debug 5: Valid Stripes", winW, winH);
    }
}

bool ZebraDetector::LoadModel(const string& modelPath)
{
    model = StatModel::load<ANN_MLP>(modelPath);
    if (model.empty()) {
        cout << "Error: Could not read the classifier " << modelPath << "\n";
        return false;
    }
    cout << "Classifier " << modelPath << " loaded.\n";
    return true;
}

// ExtractFeature: 120 col-sum + 40 row-sum = 160 features
// à¸•à¹‰à¸­à¸‡à¸•à¸£à¸‡à¸à¸±à¸š train.cpp à¸—à¸µà¹ˆà¹ƒà¸Šà¹‰ in_attributes = 160
Mat ZebraDetector::ExtractFeature(const Mat& cropImage)
{
    Mat resizedImg;
    resize(cropImage, resizedImg, Size(120, 40));

    Mat greyMat;
    if (resizedImg.channels() > 1)
        cvtColor(resizedImg, greyMat, COLOR_BGR2GRAY);
    else
        greyMat = resizedImg.clone();

    // Column sums (120 values)
    vector<int> col_sums(greyMat.cols, 0);
    float max_col = 1.0f;
    for (int i = 0; i < greyMat.cols; i++) {
        for (int k = 0; k < greyMat.rows; k++)
            col_sums[i] += greyMat.at<unsigned char>(k, i);
        if ((float)col_sums[i] > max_col) max_col = (float)col_sums[i];
    }

    // Row sums (40 values)
    vector<int> row_sums(greyMat.rows, 0);
    float max_row = 1.0f;
    for (int k = 0; k < greyMat.rows; k++) {
        for (int i = 0; i < greyMat.cols; i++)
            row_sums[k] += greyMat.at<unsigned char>(k, i);
        if ((float)row_sums[k] > max_row) max_row = (float)row_sums[k];
    }

    // Feature vector: 120 col + 40 row = 160 à¸„à¹ˆà¸²
    int total = greyMat.cols + greyMat.rows; // 160
    Mat data(1, total, CV_32F);
    for (int i = 0; i < greyMat.cols; i++)
        data.at<float>(0, i) = (float)col_sums[i] / max_col;
    for (int k = 0; k < greyMat.rows; k++)
        data.at<float>(0, greyMat.cols + k) = (float)row_sums[k] / max_row;

    return data;
}

// Helper: à¸«à¸²à¸„à¹ˆà¸²à¹€à¸‰à¸¥à¸µà¹ˆà¸¢à¸‚à¸­à¸‡ Rect à¸ˆà¸²à¸ deque (à¸ªà¸³à¸«à¸£à¸±à¸š Temporal Smoothing)
static Rect AverageRect(const deque<Rect>& rects)
{
    long long x = 0, y = 0, r = 0, b = 0;
    for (const auto& rc : rects) {
        x += rc.x; y += rc.y;
        r += rc.x + rc.width;
        b += rc.y + rc.height;
    }
    int n = (int)rects.size();
    if (n == 0) return Rect(0, 0, 0, 0);
    int ax = (int)(x / n), ay = (int)(y / n);
    int ar = (int)(r / n), ab = (int)(b / n);
    return Rect(ax, ay, ar - ax, ab - ay);
}

// ANN predict
bool ZebraDetector::TryDetectZebra(const Mat& workFrame,
    const vector<Rect>& validStripes,
    Rect& outRect)
{
    // à¸•à¹‰à¸­à¸‡à¸¡à¸µà¸‹à¸µà¹ˆ >= 3
    if ((int)validStripes.size() < 3) return false;

    // à¹à¸•à¹ˆà¸¥à¸°à¸‹à¸µà¹ˆà¸à¸§à¹‰à¸²à¸‡ >= 15% à¸ˆà¸­ à¸£à¸­à¸‡à¸£à¸±à¸šà¸—à¸²à¸‡à¸¡à¹‰à¸²à¸¥à¸²à¸¢à¹„à¸à¸¥/à¹€à¸‰à¸µà¸¢à¸‡)
    vector<Rect> wideStripes;
    for (const auto& s : validStripes)
        if (s.width >= workFrame.cols * 0.15)
            wideStripes.push_back(s);
    if ((int)wideStripes.size() < 3) return false;

    // à¸„à¸§à¸²à¸¡à¸à¸§à¹‰à¸²à¸‡à¸‹à¸µà¹ˆà¸ªà¸¡à¹ˆà¸³à¹€à¸ªà¸¡à¸­à¸à¸±à¸™ (70% à¸‚à¸­à¸‡à¸‹à¸µà¹ˆà¸•à¹‰à¸­à¸‡à¸­à¸¢à¸¹à¹ˆà¹ƒà¸™ Â±50% à¸‚à¸­à¸‡à¸„à¹ˆà¸²à¹€à¸‰à¸¥à¸µà¹ˆà¸¢)
    float totalW = 0;
    for (const auto& s : wideStripes) totalW += (float)s.width;
    float avgW = totalW / (float)wideStripes.size();
    int passSimilar = 0;
    for (const auto& s : wideStripes)
        if (fabsf((float)s.width - avgW) < avgW * 0.5f) passSimilar++;
    if (passSimilar < (int)((float)wideStripes.size() * 0.7f)) return false;

    // à¸‹à¸µà¹ˆà¸•à¹‰à¸­à¸‡à¸à¸£à¸°à¸ˆà¸²à¸¢à¹ƒà¸™ Y à¸¡à¸²à¸à¸à¸§à¹ˆà¸² 2.5x à¸„à¸§à¸²à¸¡à¸ªà¸¹à¸‡à¹€à¸‰à¸¥à¸µà¹ˆà¸¢à¸‚à¸­à¸‡à¸‹à¸µà¹ˆ
    int y_min = workFrame.rows, y_max = 0;
    float avgH = 0;
    for (const auto& s : wideStripes) {
        y_min = min(y_min, s.y);
        y_max = max(y_max, s.y + s.height);
        avgH += (float)s.height;
    }
    avgH /= (float)wideStripes.size();
    if ((float)(y_max - y_min) < avgH * 2.5f) return false;

    // à¸£à¸°à¸¢à¸°à¸«à¹ˆà¸²à¸‡à¸ªà¸¡à¸”à¸¸à¸¥ (Balanced Spacing / Y-Axis Gap) 
    // à¹€à¸£à¸µà¸¢à¸‡à¸‹à¸µà¹ˆà¸¡à¹‰à¸²à¸¥à¸²à¸¢à¸ˆà¸²à¸à¸šà¸™à¸¥à¸‡à¸¥à¹ˆà¸²à¸‡à¸¢à¸­à¸”à¹€à¸™à¸´à¸™à¹à¸à¸™ Y (à¸ˆà¸²à¸à¸£à¸°à¸¢à¸°à¹„à¸à¸¥à¸¡à¸²à¸£à¸°à¸¢à¸°à¹ƒà¸à¸¥à¹‰)
    vector<Rect> sortedStripes = wideStripes;
    sort(sortedStripes.begin(), sortedStripes.end(), [](const Rect& a, const Rect& b) {
        return a.y < b.y;
        });

    float avgStripeH = 0;
    for (const auto& s : sortedStripes) avgStripeH += s.height;
    avgStripeH /= sortedStripes.size();

    vector<int> gaps;
    int validGaps = 0;
    for (size_t i = 0; i < sortedStripes.size() - 1; i++) {
        int gap = sortedStripes[i + 1].y - (sortedStripes[i].y + sortedStripes[i].height);
        gaps.push_back(gap);
        // à¸£à¸°à¸¢à¸°à¸«à¹ˆà¸²à¸‡à¸•à¹‰à¸­à¸‡à¸à¸§à¹‰à¸²à¸‡à¸žà¸­à¸ªà¸¡à¸„à¸§à¸£ à¹„à¸¡à¹ˆà¸Šà¸´à¸”à¸•à¸´à¸”à¸à¸±à¸™à¹€à¸›à¹‡à¸™à¸‚à¸­à¸šà¸Ÿà¸¸à¸•à¸šà¸²à¸—
        if (gap > avgStripeH * 0.2f) validGaps++;
    }

    // à¹€à¸Šà¹‡à¸„à¸‚à¸±à¹‰à¸™à¸šà¸±à¸™à¹„à¸”
    if (sortedStripes.size() > 3 && validGaps < ((int)gaps.size() / 2)) return false;

    // à¹€à¸Šà¹‡à¸„à¸„à¸§à¸²à¸¡à¸ªà¸¡à¸”à¸¸à¸¥à¸‚à¸­à¸‡à¸„à¸§à¸²à¸¡à¸–à¸µà¹ˆà¹à¸¥à¸°à¸ˆà¸±à¸‡à¸«à¸§à¸° (Periodic Pattern)
    if (gaps.size() >= 2) {
        // à¸«à¸¥à¸±à¸ Perspective "à¸¢à¸´à¹ˆà¸‡à¹„à¸à¸¥à¸£à¸¹à¸›à¸¢à¸´à¹ˆà¸‡à¹€à¸¥à¹‡à¸ Gapà¸¢à¸´à¹ˆà¸‡à¹à¸„à¸š" "à¸¢à¸´à¹ˆà¸‡à¹ƒà¸à¸¥à¹‰à¸£à¸¹à¸›à¸¢à¸´à¹ˆà¸‡à¹ƒà¸«à¸à¹ˆ Gapà¸¢à¸´à¹ˆà¸‡à¸à¸§à¹‰à¸²à¸‡"
        // à¸à¸Ž: "à¸œà¸¥à¸£à¸§à¸¡Gapà¸à¸±à¹ˆà¸‡à¹„à¸à¸¥ (à¸„à¸£à¸¶à¹ˆà¸‡à¸šà¸™) à¸ˆà¸°à¸•à¹‰à¸­à¸‡à¹„à¸¡à¹ˆà¸à¸§à¹‰à¸²à¸‡à¸à¸§à¹ˆà¸² Gap à¸à¸±à¹ˆà¸‡à¹ƒà¸à¸¥à¹‰ (à¸„à¸£à¸¶à¹ˆà¸‡à¸¥à¹ˆà¸²à¸‡) à¹€à¸”à¹‡à¸”à¸‚à¸²à¸”!"
        int topHalfGapSum = 0, bottomHalfGapSum = 0;
        int half = gaps.size() / 2;
        for (int i = 0; i < half; i++) topHalfGapSum += gaps[i];
        for (int i = gaps.size() - half; i < gaps.size(); i++) bottomHalfGapSum += gaps[i];

        // à¸–à¹‰à¸²à¸£à¸°à¸¢à¸°à¸«à¹ˆà¸²à¸‡à¸Šà¹ˆà¸§à¸‡à¹„à¸à¸¥ à¸”à¸±à¸™à¸à¸§à¹‰à¸²à¸‡à¸à¸§à¹ˆà¸²à¸Šà¹ˆà¸§à¸‡à¹ƒà¸à¸¥à¹‰à¸œà¸´à¸”à¸§à¸´à¸ªà¸±à¸¢ (à¹€à¸Šà¹ˆà¸™à¹€à¸à¸´à¸”à¸ˆà¸²à¸à¹€à¸¨à¸©à¸£à¸­à¸¢à¹à¸•à¸à¸šà¸™à¸–à¸™à¸™) à¹ƒà¸«à¹‰à¸—à¸´à¹‰à¸‡
        if (topHalfGapSum > bottomHalfGapSum * 1.5f) return false;
    }

    // Merge: à¸«à¸² bounding rect à¸£à¸§à¸¡à¸‹à¸µà¹ˆà¸—à¸±à¹‰à¸‡à¸«à¸¡à¸” + padding
    int min_x = workFrame.cols, min_y = workFrame.rows;
    int max_x = 0, max_y = 0;
    for (const auto& s : wideStripes) {
        min_x = min(min_x, s.x);              min_y = min(min_y, s.y);
        max_x = max(max_x, s.x + s.width);    max_y = max(max_y, s.y + s.height);
    }
    const int padding = 15;
    Rect br;
    br.x = max(0, min_x - padding);
    br.y = max(0, min_y - padding);
    br.width = min(workFrame.cols - br.x, (max_x - min_x) + 2 * padding);
    br.height = min(workFrame.rows - br.y, (max_y - min_y) + 2 * padding);
    if (br.width <= 0 || br.height <= 0) return false;

    // ANN Predict
    Mat cropImg = workFrame(br).clone();
    Mat featureData = ExtractFeature(cropImg);
    Mat predict_responses;
    model->predict(featureData, predict_responses);

    float max_prob = predict_responses.at<float>(0, 0);
    int   max_class = 0;
    for (int c = 1; c < predict_responses.cols; c++) {
        if (predict_responses.at<float>(0, c) > max_prob) {
            max_prob = predict_responses.at<float>(0, c);
            max_class = c;
        }
    }
    if (max_class != 1) return false;

    outRect = br;
    return true;
}

// DetectAndDraw
bool ZebraDetector::DetectAndDraw(Mat& frame)
{
    // [แก้ไขจุดที่ 1] เปลี่ยนจาก return; เฉยๆ เป็น return false;
    if (frame.empty() || model.empty()) return false;

    Mat workFrame = frame;  // อ้างอิง ไม่ copy

    Mat lab, thresh;

    // LAB (Color Space) 
    cvtColor(workFrame, lab, COLOR_BGR2Lab);
    vector<Mat> channels;
    split(lab, channels);
    Mat l_channel = channels[0];

    // CLAHE (Local Contrast)
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    clahe->apply(l_channel, l_channel);

    if (debugMode) imshow("Debug 1: L-Channel + CLAHE", l_channel);

    // Median Blur
    medianBlur(l_channel, l_channel, 7);

    // Adaptive Thresholding
    adaptiveThreshold(l_channel, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY, 101, -12);

    // Trapezoid ROI mask
    {
        Mat mask = Mat::zeros(frame.size(), CV_8U);
        vector<Point> roi_pts = {
            { 0,              frame.rows },
            { frame.cols,     frame.rows },
            { (int)(frame.cols * 0.80f), (int)(frame.rows * 0.20f) },
            { (int)(frame.cols * 0.20f), (int)(frame.rows * 0.20f) }
        };
        vector<vector<Point>> polys = { roi_pts };
        fillPoly(mask, polys, Scalar(255));
        bitwise_and(thresh, mask, thresh);

        if (debugMode) {
            Mat roiViz = frame.clone();
            for (int i = 0; i < (int)roi_pts.size(); i++)
                line(roiViz, roi_pts[i], roi_pts[(i + 1) % roi_pts.size()],
                    Scalar(0, 0, 255), 2);
            Mat overlay = roiViz.clone();
            fillPoly(overlay, polys, Scalar(255, 100, 0));
            addWeighted(overlay, 0.2, roiViz, 0.8, 0, roiViz);
            for (int i = 0; i < (int)roi_pts.size(); i++)
                line(roiViz, roi_pts[i], roi_pts[(i + 1) % roi_pts.size()],
                    Scalar(0, 0, 255), 2);
            imshow("Debug 2: ROI Trapezoid Mask", roiViz);
        }
    }

    if (debugMode) imshow("Debug 3: Threshold", thresh);

    // Morphological Open & Close
    Mat kernelOpen = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(thresh, thresh, MORPH_OPEN, kernelOpen);

    Mat kernelClose = getStructuringElement(MORPH_RECT, Size(31, 5));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernelClose);

    // Find Contours & Filter Stripes
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(thresh, contours, -1, Scalar(255), FILLED);

    if (debugMode) imshow("Debug 4: Morph Close & Fill", thresh);

    vector<Rect> validStripes;
    double screen_area = workFrame.rows * workFrame.cols;

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        Rect   rect = boundingRect(contour);

        if (area > 100 && area < screen_area * 0.15) {
            float ar = (float)rect.width / (float)rect.height;
            if (ar > 1.5f && ar < 20.0f && rect.y > frame.rows / 6) {
                Mat stripeROI = thresh(rect);
                int  whitePixels = countNonZero(stripeROI);
                int  totalPixels = rect.width * rect.height;
                float density = (float)whitePixels / (float)totalPixels;

                vector<Point> hull;
                convexHull(contour, hull);
                double hull_area = contourArea(hull);
                double solidity = (hull_area > 0) ? (area / hull_area) : 0;

                if (density > 0.55f && solidity > 0.65f) {
                    validStripes.push_back(rect);
                }
            }
        }
    }

    if (debugMode) {
        Mat stripeViz = workFrame.clone();
        for (const auto& s : validStripes)
            rectangle(stripeViz, s, Scalar(255, 80, 0), 2);
        putText(stripeViz, "Valid Stripes: " + to_string(validStripes.size()),
            Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 80, 0), 2);
        imshow("Debug 5: Valid Stripes", stripeViz);
    }

    // Detect Zebra Crosswalk
    Rect detectedRect;
    bool found = TryDetectZebra(workFrame, validStripes, detectedRect);

    // Temporal Smoothing
    if (found && detectedRect.width > 0 && detectedRect.height > 0)
        detectionHistory.push_back(detectedRect);
    else
        detectionHistory.push_back(Rect(0, 0, 0, 0));

    while ((int)detectionHistory.size() > historySize)
        detectionHistory.pop_front();

    int detectCount = 0;
    deque<Rect> validHistory;
    for (const auto& r : detectionHistory) {
        if (r.width > 0) { detectCount++; validHistory.push_back(r); }
    }

    // วาดกรอบเฉลี่ยก็ต่อเมื่อพบ >= minDetectCount frame ใน history
    if (detectCount >= minDetectCount) {
        Rect drawRect = AverageRect(validHistory);
        if (drawRect.width > 0 && drawRect.height > 0) {
            rectangle(frame, drawRect, Scalar(0, 255, 0), 4);
            putText(frame, "Zebra Crossing",
                Point(drawRect.x, max(10, drawRect.y - 10)),
                FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 3);

            return true; // [แก้ไขจุดที่ 2] ถ้าวาดกรอบสำเร็จ ให้ส่งค่าว่า "เจอ"
        }
    }

    return false; // [แก้ไขจุดที่ 3] ถ้าหลุดมาถึงตรงนี้แปลว่า "ไม่เจอ"
}
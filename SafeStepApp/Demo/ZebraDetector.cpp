#include "ZebraDetector.h"
#include <vector>
#include <algorithm>
#include <climits>

using namespace cv;
using namespace std;
using namespace cv::ml;

// ===========================================================================
ZebraDetector::ZebraDetector()  {}
ZebraDetector::~ZebraDetector() {}

void ZebraDetector::SetDebugMode(bool enabled) {
    debugMode = enabled;
    if (debugMode) {
        // [FIX] ตั้งค่าหน้าต่าง Debug ให้ย่อ-ขยายได้ และกำหนดขนาดเริ่มต้นให้เล็กพอมองเห็นพร้อมกัน
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

// ===========================================================================
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

// ===========================================================================
// [FIX 6] ExtractFeature: 120 col-sum + 40 row-sum = 160 features
// ต้องตรงกับ train.cpp ที่ใช้ in_attributes = 160
Mat ZebraDetector::ExtractFeature(const Mat& cropImage)
{
    Mat resizedImg;
    resize(cropImage, resizedImg, Size(120, 40));

    Mat greyMat;
    if (resizedImg.channels() > 1)
        cvtColor(resizedImg, greyMat, COLOR_BGR2GRAY);
    else
        greyMat = resizedImg.clone();

    // --- Column sums (120 values) ---
    vector<int> col_sums(greyMat.cols, 0);
    float max_col = 1.0f;
    for (int i = 0; i < greyMat.cols; i++) {
        for (int k = 0; k < greyMat.rows; k++)
            col_sums[i] += greyMat.at<unsigned char>(k, i);
        if ((float)col_sums[i] > max_col) max_col = (float)col_sums[i];
    }

    // --- Row sums (40 values) ---
    vector<int> row_sums(greyMat.rows, 0);
    float max_row = 1.0f;
    for (int k = 0; k < greyMat.rows; k++) {
        for (int i = 0; i < greyMat.cols; i++)
            row_sums[k] += greyMat.at<unsigned char>(k, i);
        if ((float)row_sums[k] > max_row) max_row = (float)row_sums[k];
    }

    // Feature vector: 120 col + 40 row = 160 ค่า
    int total = greyMat.cols + greyMat.rows; // 160
    Mat data(1, total, CV_32F);
    for (int i = 0; i < greyMat.cols; i++)
        data.at<float>(0, i) = (float)col_sums[i] / max_col;
    for (int k = 0; k < greyMat.rows; k++)
        data.at<float>(0, greyMat.cols + k) = (float)row_sums[k] / max_row;

    return data;
}

// ===========================================================================
// Helper: หาค่าเฉลี่ยของ Rect จาก deque (สำหรับ Temporal Smoothing)
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
    int ax = (int)(x/n), ay = (int)(y/n);
    int ar = (int)(r/n), ab = (int)(b/n);
    return Rect(ax, ay, ar - ax, ab - ay);
}

// ===========================================================================
// ด่านกรองซี่ 4 ด่าน + ANN predict
bool ZebraDetector::TryDetectZebra(const Mat& workFrame,
                                   const vector<Rect>& validStripes,
                                   Rect& outRect)
{
    // ด่าน 1: ต้องมีซี่ >= 3
    if ((int)validStripes.size() < 3) return false;

    // ด่าน 2: [FIX 2] แต่ละซี่กว้าง >= 15% จอ (ลดจาก 30% → รองรับทางม้าลายไกล/เฉียง)
    vector<Rect> wideStripes;
    for (const auto& s : validStripes)
        if (s.width >= workFrame.cols * 0.15)
            wideStripes.push_back(s);
    if ((int)wideStripes.size() < 3) return false;

    // ด่าน 3: ความกว้างซี่สม่ำเสมอกัน (70% ของซี่ต้องอยู่ใน ±50% ของค่าเฉลี่ย)
    float totalW = 0;
    for (const auto& s : wideStripes) totalW += (float)s.width;
    float avgW = totalW / (float)wideStripes.size();
    int passSimilar = 0;
    for (const auto& s : wideStripes)
        if (fabsf((float)s.width - avgW) < avgW * 0.5f) passSimilar++;
    if (passSimilar < (int)((float)wideStripes.size() * 0.7f)) return false;

    // ด่าน 4: ซี่ต้องกระจายใน Y มากกว่า 2.5x ความสูงเฉลี่ยของซี่
    int y_min = workFrame.rows, y_max = 0;
    float avgH = 0;
    for (const auto& s : wideStripes) {
        y_min = min(y_min, s.y);
        y_max = max(y_max, s.y + s.height);
        avgH += (float)s.height;
    }
    avgH /= (float)wideStripes.size();
    if ((float)(y_max - y_min) < avgH * 2.5f) return false;
    // ด่าน 5: [ถูกถอดออก] กฎซ้อนทับกันแกน X ถูกเอาออก เพราะกล้องมือถืออาจจะเอียง (Roll) 
    // ถ้ากล้องเอียง ซี่ทางม้าลายจะเฉียงเป็นบันไดวน ไม่ตั้งตรงเป๊ะๆ
    // เราจะไว้ใจ "ด่าน 3: ขนาดต้องใกล้เคียงกัน" อย่างเดียวพอครับ (ถ้าขนาดพอกัน ต่อให้เยื้องกันก็ยอมให้ผ่าน)

    // ด่าน 6: [ใหม่] กฎระยะห่างสมดุล (Balanced Spacing / Y-Axis Gap) 
    // เรียงซี่ม้าลายจากบนลงล่างยอดเนินแกน Y (จากระยะไกลมาระยะใกล้)
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
        int gap = sortedStripes[i+1].y - (sortedStripes[i].y + sortedStripes[i].height);
        gaps.push_back(gap);
        // ระยะห่างต้องกว้างพอสมควร ไม่ชิดติดกันเป็นขอบฟุตบาท
        if (gap > avgStripeH * 0.2f) validGaps++;
    }
    
    // 6.1 เช็คขั้นบันได (สกัดกั้นบันไดปูน)
    if (sortedStripes.size() > 3 && validGaps < ((int)gaps.size() / 2)) return false;

    // 6.2 เช็คความสมดุลของความถี่และจังหวะ (Periodic Pattern)
    if (gaps.size() >= 2) {
        // โหมด First-Person (มุมมองแอปมือถือ): 
        // หลัก Perspective "ยิ่งไกลรูปยิ่งเล็ก Gapยิ่งแคบ" "ยิ่งใกล้รูปยิ่งใหญ่ Gapยิ่งกว้าง"
        // กฎ: "ผลรวมGapฝั่งไกล (ครึ่งบน) จะต้องไม่กว้างกว่า Gap ฝั่งใกล้ (ครึ่งล่าง) เด็ดขาด!"
        int topHalfGapSum = 0, bottomHalfGapSum = 0;
        int half = gaps.size() / 2;
        for(int i = 0; i < half; i++) topHalfGapSum += gaps[i];
        for(int i = gaps.size() - half; i < gaps.size(); i++) bottomHalfGapSum += gaps[i];
        
        // ถ้าระยะห่างช่วงไกล ดันกว้างกว่าช่วงใกล้ผิดวิสัย (เช่นเกิดจากเศษรอยแตกบนถนน) ให้ไล่ตะเพิดทิ้ง
        if (topHalfGapSum > bottomHalfGapSum * 1.5f) return false;
    }

    // Merge: หา bounding rect รวมซี่ทั้งหมด + padding
    int min_x = workFrame.cols, min_y = workFrame.rows;
    int max_x = 0,              max_y = 0;
    for (const auto& s : wideStripes) {
        min_x = min(min_x, s.x);              min_y = min(min_y, s.y);
        max_x = max(max_x, s.x + s.width);    max_y = max(max_y, s.y + s.height);
    }
    const int padding = 15;
    Rect br;
    br.x      = max(0,              min_x - padding);
    br.y      = max(0,              min_y - padding);
    br.width  = min(workFrame.cols - br.x, (max_x - min_x) + 2 * padding);
    br.height = min(workFrame.rows - br.y, (max_y - min_y) + 2 * padding);
    if (br.width <= 0 || br.height <= 0) return false;

    // ANN Predict
    Mat cropImg    = workFrame(br).clone();
    Mat featureData = ExtractFeature(cropImg);
    Mat predict_responses;
    model->predict(featureData, predict_responses);

    float max_prob  = predict_responses.at<float>(0, 0);
    int   max_class = 0;
    for (int c = 1; c < predict_responses.cols; c++) {
        if (predict_responses.at<float>(0, c) > max_prob) {
            max_prob  = predict_responses.at<float>(0, c);
            max_class = c;
        }
    }
    if (max_class != 1) return false;

    outRect = br;
    return true;
}

// ===========================================================================
void ZebraDetector::DetectAndDraw(Mat& frame)
{
    if (frame.empty() || model.empty()) return;

    Mat workFrame = frame;  // อ้างอิง ไม่ copy

    Mat lab, thresh;
    
    // [NEW FIX] 1. ลาก่อน RGB... สวัสดี LAB (Color Space) 
    // แยก "ความสว่าง" ออกจาก "สี" เด็ดขาด เพื่อลดปัญหาเงาต้นไม้/แสงแดดสะท้อน
    cvtColor(workFrame, lab, COLOR_BGR2Lab);
    vector<Mat> channels;
    split(lab, channels); 
    Mat l_channel = channels[0]; // ดึงมาเฉพาะชาแนล L (Lightness)
    
    // [EDIT] พี่เบิงอยากให้ "ไม่นำส่วนที่เป็นสีมาคิด" (เอา Color Penalty ออก)
    // ตรงนี้เราจะเลิกทำโทษ A (แดง-เขียว) และ B (น้ำเงิน-เหลือง) 
    // หมายความว่า ต่อให้เป็นทางม้าลายสีเหลือง หรือถ่ายติดกล้องเพี้ยนติดสีแดง ก็จะเห็นสว่างเท่าเดิม
    /*
    Mat a_diff, b_diff, colorPenalty;
    absdiff(channels[1], Scalar(128), a_diff); // ยิ่งเป็นสีแดง/เขียวจัด ยิ่งค่าสูง
    absdiff(channels[2], Scalar(128), b_diff); // ยิ่งเป็นสีเหลือง/น้ำเงินจัด ยิ่งค่าสูง
    add(a_diff, b_diff, colorPenalty);
    colorPenalty *= 1.5; // ขยายความรุนแรงบทลงโทษ
    subtract(l_channel, colorPenalty, l_channel);
    */

    // [NEW FIX] 2. CLAHE (Local Contrast) เร่งความคมชัดดึงทางม้าลายที่แอบในเงาออกมา
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    clahe->apply(l_channel, l_channel);

    if (debugMode) imshow("Debug 1: L-Channel + CLAHE", l_channel);

    // [NEW FIX] 3. ลบ Noise พื้นผิว (ลายกระเบื้อง, ลายแผ่นเหล็กนูนๆ) 
    // เปลี่ยนจาก GaussianBlur มาใช้ Median Blur เพราะมันเก่งเรื่องการลบรอยขรุขระ โดยที่ "ยังรักษาเส้นขอบที่คมชัดไว้ได้ (Edge Preserving)"
    // ทางม้าลายจะได้ไม่เบลอจนโดนคัดทิ้ง
    medianBlur(l_channel, l_channel, 7);

    // [NEW FIX] 4. เลิกตีขลุมด้วย Otsu -> เปลี่ยนเป็น Adaptive Thresholding
    // แก้ปัญหาภาพที่มีแดดเปรี้ยงฝั่งซ้าย และร่มเงาฝั่งขวา ให้พิจารณาแยกกันแบบจุดต่อจุด (หน้าต่างสแกน 101x101)
    // ลดความซาดิสม์ลงเหลือ -12 (สว่างกว่าเพื่อนบ้านแค่ 12 ระดับก็ผ่าน) เพราะเราใช้ Median Blur ถูจนเนียนระดับนึงแล้ว
    adaptiveThreshold(l_channel, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, 
                      THRESH_BINARY, 101, -12);

    // [FIX 3] Trapezoid ROI mask
    {
        Mat mask = Mat::zeros(frame.size(), CV_8U);
        vector<Point> roi_pts = {
            { 0,              frame.rows },
            { frame.cols,     frame.rows },
            { (int)(frame.cols * 0.80f), (int)(frame.rows * 0.20f) }, // ใช้ 0.20f สำหรับถือมือถือ
            { (int)(frame.cols * 0.20f), (int)(frame.rows * 0.20f) }
        };
        vector<vector<Point>> polys = { roi_pts };
        fillPoly(mask, polys, Scalar(255));
        bitwise_and(thresh, mask, thresh);

        // [DEBUG] วาด trapezoid ลงบน frame copy เพื่อแสดงขอบเขต ROI
        if (debugMode) {
            Mat roiViz = frame.clone();
            // วาดเส้น trapezoid สีแดง
            for (int i = 0; i < (int)roi_pts.size(); i++)
                line(roiViz, roi_pts[i], roi_pts[(i+1) % roi_pts.size()],
                     Scalar(0, 0, 255), 2);
            // เติมสีน้ำเงินโปร่งแสงใน ROI zone
            Mat overlay = roiViz.clone();
            fillPoly(overlay, polys, Scalar(255, 100, 0));
            addWeighted(overlay, 0.2, roiViz, 0.8, 0, roiViz);
            for (int i = 0; i < (int)roi_pts.size(); i++)
                line(roiViz, roi_pts[i], roi_pts[(i+1) % roi_pts.size()],
                     Scalar(0, 0, 255), 2);
            imshow("Debug 2: ROI Trapezoid Mask", roiViz);
        }
    }

    // [DEBUG] หน้าต่าง 3: Threshold
    if (debugMode) imshow("Debug 3: Threshold", thresh);

    // เปิด (Erode -> Dilate): ลบล้างรอยเชื่อมบางๆ ระหว่างซี่ตอนแสงจ้า
    Mat kernelOpen = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(thresh, thresh, MORPH_OPEN, kernelOpen);

    // ปิด (Dilate -> Erode): เชื่อมรอยแหว่งภายในซี่ตัวมันเอง
    // [NEW FIX] ปรับ Kernel แนวนอนยาวๆ (31x5) เพื่อเชื่อมขอบซ้าย-ขวาที่แหว่ง โดยไม่พาซี่บน-ล่างมาติดกัน
    Mat kernelClose = getStructuringElement(MORPH_RECT, Size(31, 5));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernelClose);

    // ─────────── Find Contours & Filter Stripes ───────────

    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // [NEW FIX] เทคนิค "อุดรูรั่ว (Fill Holes)"
    // ถมสีขาวลงไปในกรอบของวัตถุทั้งหมด เพื่อให้ซี่ที่สีซีดตรงกลางกลับมาตันเต็มก้อน (สอบผ่านด่าน Density)
    drawContours(thresh, contours, -1, Scalar(255), FILLED);

    // [DEBUG] หน้าต่าง 4: หลัง Morphological Open + Close + Fill Holes
    if (debugMode) imshow("Debug 4: Morph Close & Fill", thresh);

    vector<Rect> validStripes;
    double screen_area = workFrame.rows * workFrame.cols;

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        Rect   rect = boundingRect(contour);

        // ปรับ Area เริ่มต้นให้เล็กลง (100) สำหรับทางม้าลายที่อยู่ไกลมากๆ หรือโดนเงาตัดจนแหว่ง
        // Area ของ 1 ซี่ ไม่ควรใหญ่เกิน 15% ของหน้าจอ
        // Area ของ 1 ซี่ ไม่ควรใหญ่เกิน 15% ของหน้าจอ
        if (area > 100 && area < screen_area * 0.15) {
            float ar = (float)rect.width / (float)rect.height;
            // [STRICT FIX] ขยับ Aspect Ratio เป็น 1.5f (ต้องเป็นสี่เหลี่ยมผืนผ้าแนวนอนชัดเจน ไม่เอาทรงจตุรัสหรือแนวตั้ง)
            if (ar > 1.5f && ar < 20.0f && rect.y > frame.rows / 6) {
                
                // 1. เช็คความทึบในกรอบสี่เหลี่ยม (Density)
                Mat stripeROI = thresh(rect);
                int  whitePixels = countNonZero(stripeROI);
                int  totalPixels = rect.width * rect.height;
                float density    = (float)whitePixels / (float)totalPixels;
                
                // 2. [STRICT FIX] เช็คทรงตัน (Solidity) ป้องกันพวกรูปทรงแฉกตัว X, ตัว L หรือวงแหวน
                vector<Point> hull;
                convexHull(contour, hull);
                double hull_area = contourArea(hull);
                double solidity = (hull_area > 0) ? (area / hull_area) : 0;
                
                // บังคับให้เป็นก้อนทึบ (Density > 55%) และรูปร่างต้องไม่มีส่วนเว้าแหว่งมากเกินไป (Solidity > 65%)
                if (density > 0.55f && solidity > 0.65f) {
                    validStripes.push_back(rect);
                }
            }
        }
    }

    // [DEBUG] หน้าต่าง 5: Valid Stripes (กรอบสีน้ำเงิน) บน workFrame
    // แสดงซี่ที่ผ่านเกณฑ์ area/aspect/Y แต่ยังไม่ผ่านด่านกรอง 1-4
    if (debugMode) {
        Mat stripeViz = workFrame.clone();
        for (const auto& s : validStripes)
            rectangle(stripeViz, s, Scalar(255, 80, 0), 2);  // สีน้ำเงิน = valid stripe
        putText(stripeViz,
                "Valid Stripes: " + to_string(validStripes.size()),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 80, 0), 2);
        imshow("Debug 5: Valid Stripes", stripeViz);
    }

    // ─────────── Detect Zebra ───────────

    Rect detectedInWork;
    bool found = TryDetectZebra(workFrame, validStripes, detectedInWork);

    Rect detectedRect;
    if (found) {
        detectedRect = detectedInWork;
    }

    // ─────────── [FIX 4] Temporal Smoothing ───────────

    // push ประวัติ frame นี้ (sentinel Rect(0,0,0,0) = ไม่พบ)
    if (found && detectedRect.width > 0 && detectedRect.height > 0)
        detectionHistory.push_back(detectedRect);
    else
        detectionHistory.push_back(Rect(0, 0, 0, 0));

    // ลบ frame เก่าสุดออกถ้าเกิน historySize
    while ((int)detectionHistory.size() > historySize)
        detectionHistory.pop_front();

    // นับว่า detect จริงกี่ frame และ collect Rect ที่ valid
    int         detectCount = 0;
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
        }
    }
}

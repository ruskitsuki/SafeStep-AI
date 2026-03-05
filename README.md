# SafeStep-AI 🦺

ระบบ AI สำหรับช่วยเหลือผู้พิการทางสายตาในการข้ามถนน โดยตรวจจับทางม้าลายและแจ้งเตือนด้วยเสียง

---

## 📁 โครงสร้างโปรเจกต์

```
SafeStep-AI/
├── ZebraCrossing/          ← ระบบตรวจจับทางม้าลาย (ฝ่าย ML)
│   ├── main_augment.cpp    Phase 1: สร้าง Dataset ด้วย Augmentation
│   ├── train.cpp           Phase 3: Train ANN → ZebraModel.xml
│   ├── ZebraDetector.h     API ส่งมอบให้ทีม
│   ├── ZebraDetector.cpp   Implementation ของ Detector
│   ├── main_demo.cpp       ทดสอบ Real-time (กล้อง / วิดีโอ)
│   ├── ZebraModel.xml      โมเดลที่ผ่านการ Train แล้ว
│   └── dataset/            รูปภาพ Augmented (ไม่ถูก commit)
│       ├── zebra/          100 รูปทางม้าลาย
│       └── road/           100 รูปถนนปกติ
│
└── SafeStepApp/            ← แอปหลัก (ฝ่ายเสียงแจ้งเตือน)
```

---

## ⚙️ Tech Stack

| ส่วน | เทคโนโลยี |
|------|-----------|
| ภาษา | C++ (OOP) |
| Computer Vision | OpenCV 4.x |
| Machine Learning | `cv::ml::ANN_MLP` |
| IDE | Microsoft Visual Studio 2019/2022 |

---

## 🚀 วิธีติดตั้งและรัน

### 1. ติดตั้ง OpenCV

1. ดาวน์โหลด OpenCV 4.x จาก [opencv.org/releases](https://opencv.org/releases/)
2. แตกไฟล์ไปที่ `C:\opencv\`
3. ตั้งค่า Environment Variable: `OPENCV_DIR` = `C:\opencv\build`
4. เพิ่ม PATH: `C:\opencv\build\x64\vc16\bin`

### 2. ตั้งค่า Visual Studio (ทุก Project)

| Property | Value |
|----------|-------|
| Additional Include Dirs | `$(OPENCV_DIR)\include` |
| Additional Library Dirs | `$(OPENCV_DIR)\x64\vc16\lib` |
| Additional Dependencies | `opencv_world4xx.lib` |
| C++ Standard | C++17 |

---

## 📋 ลำดับการรัน (ฝ่าย ML)

```
Step 1 — สร้าง Dataset
  วาง zebra_src.jpg และ road_src.jpg ไว้ในโฟลเดอร์ exe
  รัน main_augment.cpp
  → ได้ dataset/zebra/ (100 รูป) และ dataset/road/ (100 รูป)

Step 2 — Train โมเดล
  รัน train.cpp
  → ได้ ZebraModel.xml

Step 3 — ทดสอบ
  รัน main_demo.cpp  (ต้องมี ZebraModel.xml ในโฟลเดอร์ exe)
  กด 'q' หรือ ESC เพื่อออก
  รันกับวิดีโอ: demo.exe path/to/video.mp4
```

---

## 🤝 Integration API (สำหรับทีมเสียงแจ้งเตือน)

เพิ่มไฟล์ `ZebraDetector.h` และ `ZebraDetector.cpp` เข้าโปรเจกต์ แล้วใช้งาน:

```cpp
#include "ZebraDetector.h"

ZebraDetector detector("ZebraModel.xml");

// ใน video loop:
if (detector.DetectAndDraw(frame)) {
    // พบทางม้าลาย → เติมโค้ดเสียงแจ้งเตือนตรงนี้
    playWarningSound();
}
```

---

## 🔧 ปรับแต่งค่า (ถ้า Accuracy ต่ำ)

| Parameter | ไฟล์ | ค่าเริ่มต้น | คำแนะนำ |
|-----------|------|------------|---------|
| `HIDDEN_NODES` | `train.cpp` | 64 | ลอง 32 หรือ 128 |
| `MAX_ITER` | `train.cpp` | 3000 | เพิ่มเป็น 5000 |
| `NUM_IMAGES` | `main_augment.cpp` | 100 | เพิ่มเป็น 200 |
| `annThreshold` | `ZebraDetector.h` | 0.4 | ลดถ้า miss บ่อย |
| `minContourArea` | `ZebraDetector.h` | 3000 | ลดถ้ากรอบไม่โชว์ |

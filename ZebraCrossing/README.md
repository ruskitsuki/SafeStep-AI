# Zebra Crossing Detection — Visual Studio Setup Guide

## โครงสร้างไฟล์

```
ZebraCrossing/
├── main_augment.cpp   ← Phase 1: สร้าง dataset (project แยก)
├── train.cpp          ← Phase 3: เทรน ANN (project แยก)
├── ZebraDetector.h    ← ส่งมอบให้ทีม
├── ZebraDetector.cpp  ← ส่งมอบให้ทีม
├── main_demo.cpp      ← ทดสอบ DetectAndDraw
├── ZebraModel.xml     ← output จากการเทรน
└── dataset/
    ├── zebra/         ← 100 รูป (สร้างโดย main_augment)
    └── road/          ← 100 รูป (สร้างโดย main_augment)
```

---

## ขั้นตอนตั้งค่า Visual Studio

> ต้องสร้าง **3 Project** แยกกัน ใน Solution เดียว

### A. ตั้งค่า OpenCV (ทำครั้งเดียว)

1. ดาวน์โหลด OpenCV 4.x จาก https://opencv.org/releases/
2. แตกไฟล์ เช่น `C:\opencv\`
3. เพิ่ม Environment Variable:
   - `OPENCV_DIR` = `C:\opencv\build`
4. เพิ่ม Path: `C:\opencv\build\x64\vc16\bin`

### B. สร้าง Project: `Augment`

| Setting | Value |
|---------|-------|
| Project Type | Console Application (C++) |
| Source Files | `main_augment.cpp` |
| Additional Includes | `$(OPENCV_DIR)\include` |
| Additional Lib Dirs | `$(OPENCV_DIR)\x64\vc16\lib` |
| Additional Deps | `opencv_world4xx.lib` (debug: `opencv_world4xxd.lib`) |
| C++ Standard | C++17 (เพื่อใช้ `std::filesystem`) |

**วิธีรัน:**
1. วาง `zebra_src.jpg` และ `road_src.jpg` ในโฟลเดอร์ที่ exe อยู่
2. รัน → จะได้ `dataset/zebra/` และ `dataset/road/`

---

### C. สร้าง Project: `Train`

| Setting | Value |
|---------|-------|
| Source Files | `train.cpp` |
| Include/Lib/Dep | เหมือน Project Augment |
| C++ Standard | C++17 |

**วิธีรัน:**
1. ต้องมี `dataset/` อยู่ในโฟลเดอร์ exe ก่อน
2. รัน → จะได้ `ZebraModel.xml`

---

### D. สร้าง Project: `Demo` (ส่งมอบให้ทีม)

| Setting | Value |
|---------|-------|
| Source Files | `ZebraDetector.h`, `ZebraDetector.cpp`, `main_demo.cpp` |
| Include/Lib/Dep | เหมือน Project Augment |

**วิธีรัน:**
1. ต้องมี `ZebraModel.xml` ในโฟลเดอร์ exe
2. รันแบบใช้กล้อง: รัน exe ตรงๆ
3. รันแบบวิดีโอ: `demo.exe path/to/video.mp4`

---

## ลำดับการทำงาน

```
1. [Augment Project] → รัน main_augment.cpp
        ↓ ได้ dataset/zebra/ + dataset/road/

2. [Train Project]   → รัน train.cpp
        ↓ ได้ ZebraModel.xml

3. [Demo Project]    → รัน main_demo.cpp
        ↓ ทดสอบ DetectAndDraw กับกล้อง/วิดีโอ
```

---

## การ Integrate กับโปรเจกต์เพื่อน

เพื่อน (ฝ่ายเสียงแจ้งเตือน) ต้องการเพียง:
1. เพิ่ม `ZebraDetector.h` และ `ZebraDetector.cpp` เข้าโปรเจกต์
2. วาง `ZebraModel.xml` ในโฟลเดอร์ exe
3. เรียกใช้:

```cpp
#include "ZebraDetector.h"

ZebraDetector detector("ZebraModel.xml");

// ใน video loop:
if (detector.DetectAndDraw(frame)) {
    // พบทางม้าลาย → เพื่อนเติมโค้ดเสียงแจ้งเตือนตรงนี้
    playWarningSound();
}
```

---

## ปรับแต่ง Parameter (ถ้า accuracy ต่ำ)

| Parameter | ไฟล์ | ค่าเริ่มต้น | คำแนะนำ |
|-----------|------|------------|---------|
| `HIDDEN_NODES` | train.cpp | 64 | ลอง 32 หรือ 128 |
| `MAX_ITER` | train.cpp | 3000 | เพิ่มเป็น 5000 |
| `minContourArea` | ZebraDetector.cpp | 3000 | ลดถ้ากรอบไม่โชว์ |
| `annThreshold` | ZebraDetector.h | 0.4 | ลดถ้า miss มาก |
| `NUM_IMAGES` | main_augment.cpp | 100 | เพิ่มเป็น 200 |

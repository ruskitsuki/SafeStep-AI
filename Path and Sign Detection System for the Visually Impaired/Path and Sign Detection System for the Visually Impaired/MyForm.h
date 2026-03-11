#pragma once
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <mmsystem.h>
#include "ZebraDetector.h"
#include "PathDetector.h" // [เพิ่มใหม่] โหลดโค้ดเพื่อนคนที่ 2
#include <msclr\marshal_cppstd.h> 
#pragma comment(lib, "winmm.lib")

namespace PathandSignDetectionSystemfortheVisuallyImpaired {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			lastAudioTime = System::DateTime::MinValue;
			videoFilePath = "";

			// โหลดโมเดลเพื่อนคนที่ 1
			zebraDetector = new ZebraDetector();
			zebraDetector->LoadModel("ZebraModel.xml");

			// [เพิ่มใหม่] โหลดระบบเพื่อนคนที่ 2
			pathDetector = new PathDetector();
		}

	protected:
		~MyForm()
		{
			if (components) { delete components; }
			if (zebraDetector) { delete zebraDetector; }
			if (pathDetector) { delete pathDetector; } // [เพิ่มใหม่] คืนค่า Memory
		}

	private:
		System::Drawing::Bitmap^ MatToBitmap(cv::Mat mat) {
			if (mat.empty()) return nullptr;
			cv::Mat bgra;
			cv::cvtColor(mat, bgra, cv::COLOR_BGR2BGRA);
			System::Drawing::Bitmap^ tempBmp = gcnew System::Drawing::Bitmap(
				bgra.cols, bgra.rows, bgra.step,
				System::Drawing::Imaging::PixelFormat::Format32bppArgb,
				(System::IntPtr)bgra.data);
			System::Drawing::Bitmap^ finalBmp = gcnew System::Drawing::Bitmap(tempBmp->Width, tempBmp->Height);
			System::Drawing::Graphics^ g = System::Drawing::Graphics::FromImage(finalBmp);
			g->DrawImage(tempBmp, 0, 0, tempBmp->Width, tempBmp->Height);
			delete g;
			delete tempBmp;
			return finalBmp;
		}

		void PlayAudioAlert(System::String^ soundType) {
			if (chkMute->Checked) return;
			System::TimeSpan timeSinceLastAudio = System::DateTime::Now - lastAudioTime;
			if (timeSinceLastAudio.TotalSeconds < 3.0) return;

			try {
				System::Media::SoundPlayer^ player = gcnew System::Media::SoundPlayer();

				if (soundType == "ZEBRA_CROSSING") player->SoundLocation = "zebra_crossing.wav";
				else if (soundType == "TURN_LEFT") player->SoundLocation = "turn_left.wav";
				else if (soundType == "TURN_RIGHT") player->SoundLocation = "turn_right.wav";
				// [เพิ่มใหม่] ไฟล์เสียงเตือนทางแยก
				else if (soundType == "JUNCTION") player->SoundLocation = "junction.wav";

				player->Play();
				lastAudioTime = System::DateTime::Now;
				this->Invoke(gcnew System::Action<System::String^>(this, &MyForm::UpdateAudioLabel), soundType);
			}
			catch (System::Exception^ ex) {}
		}

		void UpdateSignLabel(System::String^ text) { lblSign->Text = L"Sign: " + text; }
		void UpdatePathLabel(System::String^ text) { lblPath->Text = L"Path: " + text; } // [เพิ่มใหม่] อัปเดตป้าย Path
		void UpdateAudioLabel(System::String^ text) { lblAudio->Text = L"Last Audio: " + text; }

	private: System::Windows::Forms::PictureBox^ pictureBoxMain;
	private: System::Windows::Forms::GroupBox^ groupBox1;
	private: System::Windows::Forms::Button^ btnStart;
	private: System::Windows::Forms::Button^ btnStop;
	private: System::Windows::Forms::Button^ btnBrowse;
	private: System::Windows::Forms::GroupBox^ groupBox2;
	private: System::Windows::Forms::Label^ lblPath;
	private: System::Windows::Forms::Label^ lblSign;
	private: System::Windows::Forms::GroupBox^ groupBox3;
	private: System::Windows::Forms::CheckBox^ chkMute;
	private: System::Windows::Forms::Label^ lblAudio;
	private: System::ComponentModel::BackgroundWorker^ bgWorker;

	private: System::DateTime lastAudioTime;
	private: ZebraDetector* zebraDetector;
	private: PathDetector* pathDetector; // [เพิ่มใหม่] ตัวแทนของเพื่อนคนที่ 2
	private: System::String^ videoFilePath;

	private:
		System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			this->pictureBoxMain = (gcnew System::Windows::Forms::PictureBox());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->btnBrowse = (gcnew System::Windows::Forms::Button());
			this->btnStop = (gcnew System::Windows::Forms::Button());
			this->btnStart = (gcnew System::Windows::Forms::Button());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->lblPath = (gcnew System::Windows::Forms::Label());
			this->lblSign = (gcnew System::Windows::Forms::Label());
			this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
			this->chkMute = (gcnew System::Windows::Forms::CheckBox());
			this->lblAudio = (gcnew System::Windows::Forms::Label());
			this->bgWorker = (gcnew System::ComponentModel::BackgroundWorker());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxMain))->BeginInit();
			this->groupBox1->SuspendLayout();
			this->groupBox2->SuspendLayout();
			this->groupBox3->SuspendLayout();
			this->SuspendLayout();
			// 
			// pictureBoxMain
			// 
			this->pictureBoxMain->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
				| System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->pictureBoxMain->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pictureBoxMain->Location = System::Drawing::Point(-1, 1);
			this->pictureBoxMain->Name = L"pictureBoxMain";
			this->pictureBoxMain->Size = System::Drawing::Size(745, 544);
			this->pictureBoxMain->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBoxMain->TabIndex = 0;
			this->pictureBoxMain->TabStop = false;
			// 
			// groupBox1
			// 
			this->groupBox1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->groupBox1->Controls->Add(this->btnBrowse);
			this->groupBox1->Controls->Add(this->btnStop);
			this->groupBox1->Controls->Add(this->btnStart);
			this->groupBox1->Location = System::Drawing::Point(760, 48);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(201, 130);
			this->groupBox1->TabIndex = 1;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"SYSTEM CONTROLS";
			// 
			// btnBrowse
			// 
			this->btnBrowse->BackColor = System::Drawing::Color::LightSkyBlue;
			this->btnBrowse->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->btnBrowse->Location = System::Drawing::Point(15, 30);
			this->btnBrowse->Name = L"btnBrowse";
			this->btnBrowse->Size = System::Drawing::Size(170, 35);
			this->btnBrowse->TabIndex = 2;
			this->btnBrowse->Text = L"BROWSE FILE...";
			this->btnBrowse->UseVisualStyleBackColor = false;
			this->btnBrowse->Click += gcnew System::EventHandler(this, &MyForm::btnBrowse_Click);
			// 
			// btnStop
			// 
			this->btnStop->BackColor = System::Drawing::Color::Red;
			this->btnStop->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->btnStop->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			this->btnStop->Location = System::Drawing::Point(105, 75);
			this->btnStop->Name = L"btnStop";
			this->btnStop->Size = System::Drawing::Size(80, 35);
			this->btnStop->TabIndex = 1;
			this->btnStop->Text = L"STOP";
			this->btnStop->UseVisualStyleBackColor = false;
			this->btnStop->Click += gcnew System::EventHandler(this, &MyForm::btnStop_Click);
			// 
			// btnStart
			// 
			this->btnStart->BackColor = System::Drawing::Color::Lime;
			this->btnStart->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->btnStart->ForeColor = System::Drawing::SystemColors::ActiveCaptionText;
			this->btnStart->Location = System::Drawing::Point(15, 75);
			this->btnStart->Name = L"btnStart";
			this->btnStart->Size = System::Drawing::Size(80, 35);
			this->btnStart->TabIndex = 0;
			this->btnStart->Text = L"START";
			this->btnStart->UseVisualStyleBackColor = false;
			this->btnStart->Click += gcnew System::EventHandler(this, &MyForm::btnStart_Click);
			// 
			// groupBox2
			// 
			this->groupBox2->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->groupBox2->Controls->Add(this->lblPath);
			this->groupBox2->Controls->Add(this->lblSign);
			this->groupBox2->Location = System::Drawing::Point(750, 221);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(231, 120);
			this->groupBox2->TabIndex = 2;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"DETECTION RESULTS";
			// 
			// lblPath
			// 
			this->lblPath->AutoSize = true;
			this->lblPath->Font = (gcnew System::Drawing::Font(L"Tahoma", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->lblPath->Location = System::Drawing::Point(2, 76);
			this->lblPath->Name = L"lblPath";
			this->lblPath->Size = System::Drawing::Size(209, 24);
			this->lblPath->TabIndex = 1;
			this->lblPath->Text = L"Path: [ รอการตรวจจับ ]";
			// 
			// lblSign
			// 
			this->lblSign->AutoSize = true;
			this->lblSign->Font = (gcnew System::Drawing::Font(L"Tahoma", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->lblSign->Location = System::Drawing::Point(2, 29);
			this->lblSign->Name = L"lblSign";
			this->lblSign->Size = System::Drawing::Size(207, 24);
			this->lblSign->TabIndex = 0;
			this->lblSign->Text = L"Sign: [ รอการตรวจจับ ]";
			// 
			// groupBox3
			// 
			this->groupBox3->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->groupBox3->Controls->Add(this->chkMute);
			this->groupBox3->Controls->Add(this->lblAudio);
			this->groupBox3->Location = System::Drawing::Point(760, 388);
			this->groupBox3->Name = L"groupBox3";
			this->groupBox3->Size = System::Drawing::Size(200, 120);
			this->groupBox3->TabIndex = 3;
			this->groupBox3->TabStop = false;
			this->groupBox3->Text = L"AUDIO & STATUS";
			// 
			// chkMute
			// 
			this->chkMute->AutoSize = true;
			this->chkMute->Font = (gcnew System::Drawing::Font(L"Tahoma", 10.2F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->chkMute->Location = System::Drawing::Point(10, 78);
			this->chkMute->Name = L"chkMute";
			this->chkMute->Size = System::Drawing::Size(190, 25);
			this->chkMute->TabIndex = 1;
			this->chkMute->Text = L"Mute Audio (ปิดเสียง)";
			this->chkMute->UseVisualStyleBackColor = true;
			// 
			// lblAudio
			// 
			this->lblAudio->AutoSize = true;
			this->lblAudio->Font = (gcnew System::Drawing::Font(L"Tahoma", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->lblAudio->Location = System::Drawing::Point(6, 34);
			this->lblAudio->Name = L"lblAudio";
			this->lblAudio->Size = System::Drawing::Size(123, 24);
			this->lblAudio->TabIndex = 0;
			this->lblAudio->Text = L"Last Audio: -";
			// 
			// bgWorker
			// 
			this->bgWorker->WorkerReportsProgress = true;
			this->bgWorker->WorkerSupportsCancellation = true;
			this->bgWorker->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &MyForm::bgWorker_DoWork);
			this->bgWorker->ProgressChanged += gcnew System::ComponentModel::ProgressChangedEventHandler(this, &MyForm::bgWorker_ProgressChanged);
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(982, 542);
			this->Controls->Add(this->groupBox3);
			this->Controls->Add(this->groupBox2);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->pictureBoxMain);
			this->Name = L"MyForm";
			this->Text = L"Path & Sign Detection System";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxMain))->EndInit();
			this->groupBox1->ResumeLayout(false);
			this->groupBox2->ResumeLayout(false);
			this->groupBox2->PerformLayout();
			this->groupBox3->ResumeLayout(false);
			this->groupBox3->PerformLayout();
			this->ResumeLayout(false);

		}
#pragma endregion

	private: System::Void btnBrowse_Click(System::Object^ sender, System::EventArgs^ e) {
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Video Files|*.mp4;*.avi;*.mkv|All Files|*.*";
		ofd->Title = L"เลือกไฟล์วิดีโอเพื่อทำการทดสอบ";

		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			videoFilePath = ofd->FileName;
			MessageBox::Show(L"โหลดไฟล์สำเร็จ:\n" + ofd->FileName, L"Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
		}
	}

	private: System::Void btnStart_Click(System::Object^ sender, System::EventArgs^ e) {
		if (System::String::IsNullOrEmpty(videoFilePath)) {
			MessageBox::Show(L"กรุณากดปุ่ม BROWSE FILE... เพื่อเลือกวิดีโอก่อนกด START ครับ!", L"Warning", MessageBoxButtons::OK, MessageBoxIcon::Warning);
			return;
		}

		msclr::interop::marshal_context context;
		std::string stdFilePath = context.marshal_as<std::string>(videoFilePath);

		cv::VideoCapture tempCap(stdFilePath);
		if (!tempCap.isOpened()) {
			MessageBox::Show(L"ไม่สามารถเปิดไฟล์วิดีโอที่เลือกได้! ไฟล์อาจเสียหาย", L"Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}
		tempCap.release();

		if (!bgWorker->IsBusy) {
			bgWorker->RunWorkerAsync();
		}
	}

	private: System::Void btnStop_Click(System::Object^ sender, System::EventArgs^ e) {
		if (bgWorker->IsBusy) {
			bgWorker->CancelAsync();
		}
	}

	private: System::Void bgWorker_DoWork(System::Object^ sender, System::ComponentModel::DoWorkEventArgs^ e) {
		msclr::interop::marshal_context context;
		std::string stdFilePath = context.marshal_as<std::string>(videoFilePath);

		cv::VideoCapture cap(stdFilePath);
		if (!cap.isOpened()) return;

		double fps = cap.get(cv::CAP_PROP_FPS);
		if (fps <= 0 || fps > 120) fps = 30.0;
		int64 startTime = cv::getTickCount();
		long long frameCount = 1;

		cv::Mat frame;

		while (true) {
			if (bgWorker->CancellationPending) {
				e->Cancel = true;
				break;
			}

			double elapsedSec = (cv::getTickCount() - startTime) / cv::getTickFrequency();
			long long expectedFrame = (long long)(elapsedSec * fps);

			while (frameCount < expectedFrame) {
				if (!cap.grab()) break;
				frameCount++;
			}

			cap.read(frame);
			if (frame.empty()) break;
			frameCount++;

			if (frame.cols > 800) {
				float scale = 800.0f / frame.cols;
				cv::resize(frame, frame, cv::Size(), scale, scale);
			}

			// ==========================================================
			// 1. ตรวจจับทางม้าลาย (เพื่อนคนที่ 1)
			// ==========================================================
			bool isZebraFound = zebraDetector->DetectAndDraw(frame);
			if (isZebraFound) {
				this->Invoke(gcnew System::Action<System::String^>(this, &MyForm::UpdateSignLabel), L"[ ทางม้าลาย ]");
				PlayAudioAlert("ZEBRA_CROSSING");
			}
			else {
				this->Invoke(gcnew System::Action<System::String^>(this, &MyForm::UpdateSignLabel), L"[ ไม่พบสัญลักษณ์ ]");
			}

			// ==========================================================
			// 2. ตรวจจับเส้นทาง (เพื่อนคนที่ 2)
			// ==========================================================
			std::string pathType = pathDetector->detect(frame);
			System::String^ sysPathType = gcnew System::String(pathType.c_str());

			// อัปเดตข้อความบนหน้าจอ UI
			this->Invoke(gcnew System::Action<System::String^>(this, &MyForm::UpdatePathLabel), L"[ " + sysPathType + L" ]");

			// ถ้าเป็นทางแยก ให้เล่นเสียงเตือน
			if (pathType == "JUNCTION") {
				PlayAudioAlert("JUNCTION");
			}

			// ==========================================================
			// นำภาพส่งขึ้นจอ
			// ==========================================================
			System::Drawing::Bitmap^ bmp = MatToBitmap(frame);
			bgWorker->ReportProgress(0, bmp);

			elapsedSec = (cv::getTickCount() - startTime) / cv::getTickFrequency();
			expectedFrame = (long long)(elapsedSec * fps);
			int delayMs = 1;
			if (frameCount > expectedFrame) {
				double targetTimeSec = (double)frameCount / fps;
				delayMs = (int)((targetTimeSec - elapsedSec) * 1000.0);
				if (delayMs < 1) delayMs = 1;
			}
			System::Threading::Thread::Sleep(delayMs);
		}
	}

	private: System::Void bgWorker_ProgressChanged(System::Object^ sender, System::ComponentModel::ProgressChangedEventArgs^ e) {
		if (pictureBoxMain->Image != nullptr) {
			delete pictureBoxMain->Image;
		}
		pictureBoxMain->Image = (System::Drawing::Bitmap^)e->UserState;
	}

	};
}
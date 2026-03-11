#include "MyForm.h"

using namespace System;
using namespace System::Windows::Forms;

[STAThreadAttribute]
int main(array<String^>^ args) {
    // ตั้งค่าให้หน้าต่าง UI ดูสวยงามตามสไตล์ Windows
    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);

    // เรียกใช้งานหน้าจอ MyForm ของคุณขึ้นมา
    PathandSignDetectionSystemfortheVisuallyImpaired::MyForm form;
    Application::Run(% form);

    return 0;
}

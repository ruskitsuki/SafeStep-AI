#include "pch.h"
#include <windows.h>
#include "opencv2/opencv.hpp"
#include <shobjidl.h> 
#include <iostream>
#include <string>

int main_motion1(std::string videoPath);
int main_motion2(std::string videoPath);

std::string getFilePath() {
    std::string path = "";
    IFileOpenDialog* pFileOpen;
 
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (SUCCEEDED(hr)) {
        hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL, IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));
        if (SUCCEEDED(hr)) {
            hr = pFileOpen->Show(NULL);
            if (SUCCEEDED(hr)) {
                IShellItem* pItem;
                hr = pFileOpen->GetResult(&pItem);
                if (SUCCEEDED(hr)) {
                    PWSTR pszFilePath;
                    hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);
                    if (SUCCEEDED(hr)) {
                        std::wstring ws(pszFilePath);
                        path = std::string(ws.begin(), ws.end());
                        CoTaskMemFree(pszFilePath);
                    }
                    pItem->Release();
                }
            }
            pFileOpen->Release();
        }
        CoUninitialize();
    }
    return path;
}

int main() {
    while (true) {
        int choice;
        std::cout << "\n=== Motion Detection Main Menu ===" << std::endl;
        std::cout << "1. Running Average" << std::endl;
        std::cout << "2. Mixture of Gaussians (MOG2)" << std::endl;
        std::cout << "0. Exit Program" << std::endl;
        std::cout << "Choice: ";
        std::cin >> choice;

        if (choice == 0) break;

        if (choice != 1 && choice != 2) {
            std::cout << "Invalid choice! Please try again." << std::endl;
            continue;
        }

        std::string selectedFile = getFilePath(); 

        if (selectedFile.empty()) {
            std::cout << "No file selected! Returning to menu..." << std::endl;
            continue;
        }

        if (choice == 1) {
            main_motion1(selectedFile);
        }
        else {
            main_motion2(selectedFile);
        }

        cv::destroyAllWindows();
        std::cout << "\nVideo processing finished. Returning to main menu..." << std::endl;
    }

    std::cout << "Exiting program." << std::endl;
    return 0;
}
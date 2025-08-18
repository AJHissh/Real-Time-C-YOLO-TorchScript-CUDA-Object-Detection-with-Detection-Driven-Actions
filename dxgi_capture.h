#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_2.h> 
#include <d3d11.h>
#include <wrl/client.h>
#include <opencv2/opencv.hpp>

using Microsoft::WRL::ComPtr;

class DXCapture {
public:
    DXCapture();
    ~DXCapture();
    
    bool init();
    cv::Mat capture();
    void release();

private:
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGIOutputDuplication> duplication_;
    ComPtr<ID3D11Texture2D> staging_tex_;
    DXGI_OUTPUT_DESC output_desc_;
    bool initialized_ = false;
};

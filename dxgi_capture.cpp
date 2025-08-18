#include "dxgi_capture.h"
#include <iostream>

DXCapture::DXCapture() {
    if (!init()) {
        std::cerr << "Failed to initialize DXGI capture" << std::endl;
    }
}

DXCapture::~DXCapture() {
    release();
}

bool DXCapture::init() {
    // Create D3D11 device
    D3D_FEATURE_LEVEL feature_level;
    HRESULT hr = D3D11CreateDevice(
        nullptr, 
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &device_,
        &feature_level,
        &context_);

    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device: " << hr << std::endl;
        return false;
    }

    ComPtr<IDXGIDevice> dxgi_device;
    device_.As(&dxgi_device);

    ComPtr<IDXGIAdapter> adapter;
    dxgi_device->GetAdapter(&adapter);

    ComPtr<IDXGIOutput> output;
    adapter->EnumOutputs(0, &output);
    output->GetDesc(&output_desc_);

    ComPtr<IDXGIOutput1> output2;
    output.As(&output2);
    hr = output2->DuplicateOutput(device_.Get(), &duplication_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create duplication interface: " << hr << std::endl;
        return false;
    }

    D3D11_TEXTURE2D_DESC tex_desc = {};
    tex_desc.Width = output_desc_.DesktopCoordinates.right;
    tex_desc.Height = output_desc_.DesktopCoordinates.bottom;
    tex_desc.MipLevels = 1;
    tex_desc.ArraySize = 1;
    tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    tex_desc.SampleDesc.Count = 1;
    tex_desc.Usage = D3D11_USAGE_STAGING;
    tex_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    
    hr = device_->CreateTexture2D(&tex_desc, nullptr, &staging_tex_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: " << hr << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
}

cv::Mat DXCapture::capture() {
    if (!initialized_) return cv::Mat();

    DXGI_OUTDUPL_FRAME_INFO frame_info;
    ComPtr<IDXGIResource> resource;
    HRESULT hr = duplication_->AcquireNextFrame(100, &frame_info, &resource);
    if (FAILED(hr)) {
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            return cv::Mat(); 
        }
        std::cerr << "Failed to acquire frame: " << hr << std::endl;
        return cv::Mat();
    }

    ComPtr<ID3D11Texture2D> tex;
    resource.As(&tex);
    context_->CopyResource(staging_tex_.Get(), tex.Get());

    D3D11_MAPPED_SUBRESOURCE map;
    hr = context_->Map(staging_tex_.Get(), 0, D3D11_MAP_READ, 0, &map);
    if (FAILED(hr)) {
        duplication_->ReleaseFrame();
        std::cerr << "Failed to map texture: " << hr << std::endl;
        return cv::Mat();
    }

    cv::Mat frame(
        output_desc_.DesktopCoordinates.bottom,
        output_desc_.DesktopCoordinates.right,
        CV_8UC4,
        map.pData,
        map.RowPitch
    );

    cv::Mat frame_bgr;
    cv::cvtColor(frame, frame_bgr, cv::COLOR_BGRA2BGR);

    context_->Unmap(staging_tex_.Get(), 0);
    duplication_->ReleaseFrame();

    return frame_bgr;
}

void DXCapture::release() {
    if (duplication_) duplication_->ReleaseFrame();
    initialized_ = false;
}

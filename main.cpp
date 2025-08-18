#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/cuda.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <deque>
#include "dxgi_capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;
const int CROSSHAIR_X = SCREEN_WIDTH / 2;
const int CROSSHAIR_Y = SCREEN_HEIGHT / 2;
const float CONFIDENCE_THRESHOLD = 0.30f;
const int SCAN_AREA_SIZE = 700;
const int MODEL_SIZE = 640;

const float CENTER_MASS_Y_OFFSET = -20.0f;
const float MIN_TARGET_HEIGHT = 100.0f;

const float SMOOTHING_FACTOR = 0.50f;
const float MAX_SPEED = 100.00f;
const float PREDICTION_FACTOR = 0.25f;
const float LOCK_RADIUS = 10.0f;
const int HISTORY_SIZE = 10;

const int CENTER_RADIUS = 1; 
const float CENTER_MASS_WEIGHT = 1.0f; 
const float BBOX_CENTER_WEIGHT = 1.0f; 

const int TARGET_LOST_FRAMES_THRESHOLD = 5;
const int SINGLE_CLICK_LOCK_DURATION_MS = 50; 

struct AimState {
    bool is_locked = false;
    int current_target_id = -1;
    std::deque<std::pair<float, float>> movement_history;
    float remainder_x = 0;
    float remainder_y = 0;
    float last_target_size = 0.0f;
    int target_lost_frames = 0;
    float last_valid_x = -1;
    float last_valid_y = -1;
    std::chrono::steady_clock::time_point last_click_time;
    bool click_lock_active = false;
};


void debug_print(const std::string& message) {
    static HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(console, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
    std::cout << "[DEBUG] " << message << std::endl;
    SetConsoleTextAttribute(console, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
}

std::pair<float, float> get_average_movement(AimState& state) {
    if (state.movement_history.empty()) return {0, 0};
    
    float avg_dx = 0, avg_dy = 0;
    for (const auto& move : state.movement_history) {
        avg_dx += move.first;
        avg_dy += move.second;
    }
    avg_dx /= state.movement_history.size();
    avg_dy /= state.movement_history.size();
    
    return {avg_dx, avg_dy};
}

void move_mouse(float dx, float dy, AimState& state, int target_id, float target_height) {
    state.last_target_size = target_height;
    
    float distance_to_target = std::sqrt(dx*dx + dy*dy);
    float size_factor = std::clamp((target_height - MIN_TARGET_HEIGHT) / 50.0f, 0.2f, 1.0f);
    float effective_smoothing = SMOOTHING_FACTOR;
    
    if (distance_to_target < LOCK_RADIUS * 2) {
        size_factor = 1.0f; 
        effective_smoothing = SMOOTHING_FACTOR * 0.8f; 
    }
    
    float speed_scale = std::clamp(distance_to_target / (LOCK_RADIUS * 2), 0.3f, 1.0f);
    
    if (distance_to_target > LOCK_RADIUS * 3) {
        state.movement_history.push_back({dx, dy});
        if (state.movement_history.size() > HISTORY_SIZE) {
            state.movement_history.pop_front();
        }
        auto [avg_dx, avg_dy] = get_average_movement(state);
        dx = avg_dx * effective_smoothing + dx * (1.0f - effective_smoothing);
        dy = avg_dy * effective_smoothing + dy * (1.0f - effective_smoothing);
    }
    
    float current_speed = std::sqrt(dx * dx + dy * dy);
    float max_effective_speed = MAX_SPEED * size_factor * speed_scale;
    if (current_speed > max_effective_speed) {
        dx *= max_effective_speed / current_speed;
        dy *= max_effective_speed / current_speed;
    }
    
    float effective_lock_radius = LOCK_RADIUS * (target_height / 100.0f);
    state.is_locked = (distance_to_target <= effective_lock_radius);
    if (state.is_locked) {
        dx *= 1.5f;
        dy *= 1.5f;
        state.current_target_id = target_id;
        state.target_lost_frames = 0;
    }
    
    float total_dx = dx + state.remainder_x;
    float total_dy = dy + state.remainder_y;
    
    int move_x = static_cast<int>(total_dx);
    int move_y = static_cast<int>(total_dy);
    
    state.remainder_x = total_dx - move_x;
    state.remainder_y = total_dy - move_y;
    
    if (move_x != 0 || move_y != 0) {
        INPUT input = {0};
        input.type = INPUT_MOUSE;
        input.mi.dx = move_x;
        input.mi.dy = move_y;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        SendInput(1, &input, sizeof(INPUT));
    }
}

cv::Mat crop_frame(const cv::Mat& frame) {
    const int capture_width = SCAN_AREA_SIZE;
    const int capture_height = SCAN_AREA_SIZE;
    const int left = std::max(0, CROSSHAIR_X - capture_width / 2);
    const int top = std::max(0, CROSSHAIR_Y - capture_height / 2);
    const int width = std::min(capture_width, frame.cols - left);
    const int height = std::min(capture_height, frame.rows - top);

    return frame(cv::Rect(left, top, width, height));
}

std::vector<std::tuple<float, float, float, float, int, float, float>> run_detection(
    torch::jit::script::Module& model,
    const cv::Mat& frame,
    float& prev_target_x,
    float& prev_target_y) {

    std::vector<std::tuple<float, float, float, float, int, float, float>> targets;
    static int target_counter = 0;

    try {
        float scale = std::min(MODEL_SIZE / float(frame.cols), MODEL_SIZE / float(frame.rows));
        int pad_x = (MODEL_SIZE - frame.cols * scale) / 2;
        int pad_y = (MODEL_SIZE - frame.rows * scale) / 2;
        
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(), scale, scale);
        
        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, pad_y, pad_y, pad_x, pad_x, 
                          cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        
        torch::Tensor img_tensor = torch::from_blob(padded.data, {padded.rows, padded.cols, 3}, torch::kByte);
        img_tensor = img_tensor.permute({2,0,1})
                     .to(torch::kFloat)
                     .div(255)
                     .unsqueeze(0)
                     .to(torch::kCUDA);

        torch::NoGradGuard no_grad;
        auto output = model.forward({img_tensor}).toTensor();
        
        if (output.dim() == 0) {
            return targets; 
        }

        auto detections = output.squeeze(0);
        if (detections.size(0) == 0) {
            return targets; 
        }

        auto conf_mask = detections.index({4}) > CONFIDENCE_THRESHOLD;
        auto filtered = detections.index({torch::indexing::Slice(), conf_mask});

        if (filtered.size(1) == 0) {
            return targets; 
        }

        auto boxes = filtered.slice(0, 0, 4);  // [cx,cy,w,h] for all detections
        
        auto scale_tensor = torch::tensor(scale, torch::kFloat32).to(torch::kCUDA);
        auto pad_x_tensor = torch::tensor(pad_x, torch::kFloat32).to(torch::kCUDA);
        auto pad_y_tensor = torch::tensor(pad_y, torch::kFloat32).to(torch::kCUDA);
        auto offset_x = torch::tensor((SCREEN_WIDTH - frame.cols) / 2, torch::kFloat32).to(torch::kCUDA);
        auto offset_y = torch::tensor((SCREEN_HEIGHT - frame.rows) / 2, torch::kFloat32).to(torch::kCUDA);
        
        boxes.index({0}) = (boxes.index({0}) - pad_x_tensor) / scale_tensor + offset_x;
        boxes.index({1}) = (boxes.index({1}) - pad_y_tensor) / scale_tensor + offset_y;
        boxes.index({2}) = boxes.index({2}) / scale_tensor;
        boxes.index({3}) = boxes.index({3}) / scale_tensor;
        
        if (prev_target_x > 0 && prev_target_y > 0) {
            auto prev_x_tensor = torch::tensor(prev_target_x, torch::kFloat32).to(torch::kCUDA);
            auto prev_y_tensor = torch::tensor(prev_target_y, torch::kFloat32).to(torch::kCUDA);
            auto dx = (boxes.index({0}) - prev_x_tensor) * PREDICTION_FACTOR;
            auto dy = (boxes.index({1}) - prev_y_tensor) * PREDICTION_FACTOR;
            boxes.index({0}) += dx;
            boxes.index({1}) += dy;
        }
        
        auto center_mass_y = boxes.index({1}) + CENTER_MASS_Y_OFFSET;
        
        auto crosshair_x = torch::tensor(CROSSHAIR_X, torch::kFloat32).to(torch::kCUDA);
        auto crosshair_y = torch::tensor(CROSSHAIR_Y, torch::kFloat32).to(torch::kCUDA);
        
        auto dist_to_crosshair = torch::sqrt(
            torch::pow(boxes.index({0}) - crosshair_x, 2) + 
            torch::pow(boxes.index({1}) - crosshair_y, 2));
        
        auto dist_to_center_mass = torch::sqrt(
            torch::pow(boxes.index({0}) - crosshair_x, 2) + 
            torch::pow(center_mass_y - crosshair_y, 2));
        
        auto scores = (dist_to_center_mass * CENTER_MASS_WEIGHT) + 
                     (dist_to_crosshair * BBOX_CENTER_WEIGHT);
        
        auto scan_radius = torch::tensor(SCAN_AREA_SIZE/2, torch::kFloat32).to(torch::kCUDA);
        auto in_scan_area = dist_to_crosshair <= scan_radius;
        
        auto valid_indices = torch::nonzero(in_scan_area).squeeze(1);
        
        if (valid_indices.size(0) > 0) {
            auto valid_boxes = boxes.index({torch::indexing::Slice(), valid_indices}).to(torch::kCPU);
            auto valid_dist_crosshair = dist_to_crosshair.index({valid_indices}).to(torch::kCPU);
            auto valid_heights = boxes.index({3, valid_indices}).to(torch::kCPU);
            auto valid_dist_center_mass = dist_to_center_mass.index({valid_indices}).to(torch::kCPU);
            auto valid_scores = scores.index({valid_indices}).to(torch::kCPU);
            
            for (int i = 0; i < valid_indices.size(0); i++) {
                targets.emplace_back(
                    valid_boxes[0][i].item<float>(),
                    valid_boxes[1][i].item<float>(),
                    valid_dist_crosshair[i].item<float>(),
                    valid_heights[i].item<float>(),
                    target_counter++,
                    valid_dist_center_mass[i].item<float>(),
                    valid_scores[i].item<float>()
                );
            }
        }

    } catch (const std::exception& e) {
        debug_print("Detection error: " + std::string(e.what()));
    }

    return targets;
}

std::tuple<float, float, float, float, int> find_best_target(
    const std::vector<std::tuple<float, float, float, float, int, float, float>>& targets,
    AimState& aim_state) {
    
    if (targets.empty()) {
        aim_state.target_lost_frames++;
        if (aim_state.target_lost_frames >= TARGET_LOST_FRAMES_THRESHOLD) {
            aim_state.is_locked = false;
            aim_state.current_target_id = -1;
        }
        return std::make_tuple(-1, -1, -1, -1, -1);
    }
    
    std::vector<std::tuple<float, float, float, float, int, float, float>> center_targets;
    for (const auto& target : targets) {
        float x = std::get<0>(target);
        float y = std::get<1>(target);
        if (std::abs(x - CROSSHAIR_X) <= CENTER_RADIUS && std::abs(y - CROSSHAIR_Y) <= CENTER_RADIUS) {
            center_targets.push_back(target);
        }
    }
    
    if (!center_targets.empty()) {
        auto best_center = *std::min_element(center_targets.begin(), center_targets.end(),
            [](const auto& a, const auto& b) {
                return std::get<6>(a) < std::get<6>(b);
            });
        
        return std::make_tuple(std::get<0>(best_center), std::get<1>(best_center),
                             std::get<2>(best_center), std::get<3>(best_center),
                             std::get<4>(best_center));
    }
    
    if (aim_state.is_locked) {
        float current_score = std::numeric_limits<float>::max();
        bool locked_target_found = false;
        
        for (const auto& target : targets) {
            if (std::get<4>(target) == aim_state.current_target_id) {
                locked_target_found = true;
                current_score = std::get<6>(target);
                break;
            }
        }
        
        if (locked_target_found) {
            auto best_alternative = *std::min_element(targets.begin(), targets.end(),
                [](const auto& a, const auto& b) {
                    return std::get<6>(a) < std::get<6>(b);
                });
            
            float alternative_score = std::get<6>(best_alternative);
            
            if (alternative_score < (current_score * 0.8f)) {
                return std::make_tuple(std::get<0>(best_alternative), std::get<1>(best_alternative),
                                     std::get<2>(best_alternative), std::get<3>(best_alternative),
                                     std::get<4>(best_alternative));
            }
            
            for (const auto& target : targets) {
                if (std::get<4>(target) == aim_state.current_target_id) {
                    return std::make_tuple(std::get<0>(target), std::get<1>(target),
                                         std::get<2>(target), std::get<3>(target),
                                         std::get<4>(target));
                }
            }
        }
        
        aim_state.target_lost_frames++;
        if (aim_state.target_lost_frames >= TARGET_LOST_FRAMES_THRESHOLD) {
            aim_state.is_locked = false;
            aim_state.current_target_id = -1;
        }
        return std::make_tuple(-1, -1, -1, -1, -1);
    }
    
    auto best_target = *std::min_element(targets.begin(), targets.end(),
        [](const auto& a, const auto& b) {
            return std::get<6>(a) < std::get<6>(b);
        });
    
    return std::make_tuple(std::get<0>(best_target), std::get<1>(best_target),
                         std::get<2>(best_target), std::get<3>(best_target),
                         std::get<4>(best_target));
}

int main() {
    DXCapture dxgi_capture;
    if (!dxgi_capture.init()) {
        debug_print("Failed to initialize DXGI capture");
        return -1;
    }

    torch::jit::Module model;
    try {
        model = torch::jit::load("model.torchscript");
        model.to(torch::kCUDA);
        model.eval();
        debug_print("Model loaded successfully");
    } catch (...) {
        debug_print("Failed to load model");
        return -1;
    }

    AimState aim_state;
    float prev_target_x = -1, prev_target_y = -1;
    bool was_mouse_down = false;

    debug_print("Starting enhanced single-target aim assist - Press F5 to exit");
    debug_print("Crosshair position: " + std::to_string(CROSSHAIR_X) + "," + std::to_string(CROSSHAIR_Y));

    while (!(GetAsyncKeyState(VK_F5) & 0x8000)) {
        bool is_mouse_down = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
        
        if (was_mouse_down && !is_mouse_down) {
            aim_state.click_lock_active = true;
            aim_state.last_click_time = std::chrono::steady_clock::now();
        }
        was_mouse_down = is_mouse_down;

        bool should_process = is_mouse_down || 
                            (aim_state.click_lock_active && 
                             std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - aim_state.last_click_time).count() < SINGLE_CLICK_LOCK_DURATION_MS);

        if (should_process) {
            cv::Mat frame = dxgi_capture.capture();
            if (frame.empty()) continue;
            
            cv::Mat cropped = crop_frame(frame);
            auto targets = run_detection(model, cropped, prev_target_x, prev_target_y);
            auto best_target = find_best_target(targets, aim_state);

            if (std::get<4>(best_target) != -1) {
                auto [target_x, target_y, dist, target_height, target_id] = best_target;
                
                target_y += CENTER_MASS_Y_OFFSET;
                
                float dx = target_x - CROSSHAIR_X;
                float dy = target_y - CROSSHAIR_Y;
                
                move_mouse(dx, dy, aim_state, target_id, target_height);
                prev_target_x = target_x;
                prev_target_y = target_y;
                aim_state.last_valid_x = target_x;
                aim_state.last_valid_y = target_y;
            }
            
            if (is_mouse_down) {
                aim_state.click_lock_active = false;
            }
        } else {
            if (aim_state.click_lock_active && 
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - aim_state.last_click_time).count() >= SINGLE_CLICK_LOCK_DURATION_MS) {
                aim_state.click_lock_active = false;
            }
            
            if (!is_mouse_down && !aim_state.click_lock_active) {
                aim_state = AimState();
            }
        }
        Sleep(8);
    }
    return 0;
}

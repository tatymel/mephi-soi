#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <iomanip>


std::vector<float> generate_test_data(size_t size, float noise_level, float outlier, size_t outlier_step);
void write_array_to_file(const char* filename, const float* input_data, const float* filtered_data, size_t size);
bool compare_data(const float* A, const float* B, size_t size);

std::vector<uint8_t> generate_test_image(size_t width, size_t height, size_t stride);

std::vector<float> generate_test_data(size_t size, float noise_level, float outlier, size_t outlier_step) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-noise_level, noise_level);

    //������: ����� + ���
    for (size_t i = 0; i < size; ++i) {
        float signal = std::sin(i * 0.1f);
        float noise = dist(gen);
        //��������� �������
        if (outlier_step > 0 && i % outlier_step == 0 && i > 0) {
            float direction = (signal >= 0) ? 1.0f : -1.0f;
            noise += outlier * direction;
        }
        data[i] = signal + noise;
    }

    return data;
}

void write_array_to_file(const char* filename, const float* input_data, const float* filtered_data, size_t size) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file << "input;filtered\n";
    for (size_t i = 0; i < size; ++i) {
        file << std::fixed << std::setprecision(3) << input_data[i] << ";"
            << std::fixed << std::setprecision(3) << filtered_data[i] << "\n";
    }

    std::cout << "Data file complete" << std::endl;
}

bool compare_data(const float* A, const float* B, size_t size) {
    for (size_t i = 0; i < size; ++i)
        if (A[i] != B[i]) return false;
    return true;
}

std::vector<uint8_t> generate_test_image(size_t width, size_t height, size_t stride) {
    std::vector<uint8_t> image(height * stride);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise_dist(-20.0f, 20.0f);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t i = y * width + x;

            float signal = (std::sin(i * 0.0001f) * 0.5f + 0.5f) * 200.0f + 27.5f;
            float noise = noise_dist(gen);

            if (i % 50 == 0 && i > 0) {
                float direction = (signal >= 128.0f) ? 1.0f : -1.0f;
                noise += 80.0f * direction;
            }

            float value = signal + noise;
            value = std::max(0.0f, std::min(255.0f, value));
            image[y * stride + x] = static_cast<uint8_t>(value);
        }
    }

    return image;
}
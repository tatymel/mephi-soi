#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include "utils.h"



class MedianFilterGPU {
private:
    static float median_7(float arr[7]);
    static uint8_t median_9(uint8_t window[9]);

public:
    static void median_filter_7(const float* input, float* output, size_t length);
    static void median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride);
};

//����������� ���� �� 7 ���������
float MedianFilterGPU::median_7(float arr[7]) {
    cond_swap(arr[0], arr[6]);
    cond_swap(arr[2], arr[3]);
    cond_swap(arr[4], arr[5]);

    cond_swap(arr[0], arr[2]);
    cond_swap(arr[1], arr[4]);
    cond_swap(arr[3], arr[6]);

    arr[1] = get_max(arr[0], arr[1]);
    cond_swap(arr[2], arr[5]);
    cond_swap(arr[3], arr[4]);

    arr[2] = get_max(arr[1], arr[2]);
    arr[4] = get_min(arr[4], arr[6]);

    arr[3] = get_max(arr[2], arr[3]);
    arr[4] = get_min(arr[4], arr[5]);

    arr[3] = get_min(arr[3], arr[4]);

    return arr[3];
}

void MedianFilterGPU::median_filter_7(const float* input, float* output, size_t length) {
    sycl::queue q;

    size_t N = length;
    float* d_input = sycl::malloc_shared<float>(N, q);
    float* d_output = sycl::malloc_shared<float>(N, q);

    q.memcpy(d_input, input, N * sizeof(float)).wait();

    //����������� �������� ��������� �� ������� �������
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N - 6), [=](sycl::id<1> idx) {
            size_t i = idx[0] + 3;//������ ����� ������������ ���� ������
            float window[7];//��������� ������ ��� ������� ������

            for (int j = -3; j <= 3; ++j) window[j + 3] = d_input[i + j];

            d_output[i] = median_7(window);
        });
    });
    q.wait();

    //������� �������� ��������� �� ������� �����
    float window[7];

    //������ 3 ��������
    for (size_t i = 0; i < 3 && i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    //��������� 3 ��������
    for (size_t i = (N > 3 ? N - 3 : 0); i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    //��������� ����������
    q.memcpy(output, d_output, N * sizeof(float)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}



//-------------------------- 2D median filter 3x3 with shared memory --------------------------



uint8_t MedianFilterGPU::median_9(uint8_t window[9]) {
    cond_swap(window[0], window[3]);
    cond_swap(window[1], window[7]);
    cond_swap(window[2], window[5]);
    cond_swap(window[4], window[8]);

    cond_swap(window[0], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[8]);
    cond_swap(window[5], window[6]);

    window[2] = get_max(window[0], window[2]);
    cond_swap(window[1], window[3]);
    cond_swap(window[4], window[5]);
    window[7] = get_min(window[7], window[8]);

    window[4] = get_max(window[1], window[4]);
    window[3] = get_min(window[3], window[6]);
    window[5] = get_min(window[5], window[7]);

    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[5]);

    window[3] = get_max(window[2], window[3]);
    window[4] = get_min(window[4], window[5]);

    window[4] = get_max(window[3], window[4]);

    return window[4];
}


void MedianFilterGPU::median_filter_3x3(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride) {
    sycl::queue q;

    size_t input_size = height * stride;
    size_t output_size = height * stride;

    uint8_t* d_input = sycl::malloc_device<uint8_t>(input_size, q);
    uint8_t* d_output = sycl::malloc_device<uint8_t>(output_size, q);

    q.memcpy(d_input, input, input_size * sizeof(uint8_t)).wait();

    constexpr size_t LOCAL_WIDTH = 16;
    constexpr size_t LOCAL_HEIGHT = 16;

    constexpr size_t SHARED_WIDTH = LOCAL_WIDTH + 2;
    constexpr size_t SHARED_HEIGHT = LOCAL_HEIGHT + 2;

    size_t global_width = ((width + LOCAL_WIDTH - 1) / LOCAL_WIDTH) * LOCAL_WIDTH;
    size_t global_height = ((height + LOCAL_HEIGHT - 1) / LOCAL_HEIGHT) * LOCAL_HEIGHT;

    q.submit([&](sycl::handler& h) {
        auto local_mem = sycl::local_accessor<uint8_t, 2>(
            sycl::range<2>(SHARED_HEIGHT, SHARED_WIDTH), h);

        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(global_height, global_width),
                sycl::range<2>(LOCAL_HEIGHT, LOCAL_WIDTH)
            ),
            [=](sycl::nd_item<2> item) {
                size_t global_x = item.get_global_id(1);
                size_t global_y = item.get_global_id(0);
                size_t local_x = item.get_local_id(1);
                size_t local_y = item.get_local_id(0);

                if (global_x < width && global_y < height) {
                    local_mem[local_y + 1][local_x + 1] = d_input[global_y * stride + global_x];
                }

                if (local_x == 0) {
                    size_t load_x = (global_x > 0) ? global_x - 1 : global_x;
                    if (global_y < height && load_x < width) {
                        local_mem[local_y + 1][0] = d_input[global_y * stride + load_x];
                    }
                }
                if (local_x == LOCAL_WIDTH - 1 || local_x == item.get_local_range(1) - 1) {
                    size_t load_x = (global_x + 1 < width) ? global_x + 1 : global_x;
                    if (global_y < height && global_x < width && load_x < width) {
                        local_mem[local_y + 1][local_x + 2] = d_input[global_y * stride + load_x];
                    }
                }

                if (local_y == 0) {
                    size_t load_y = (global_y > 0) ? global_y - 1 : global_y;
                    if (global_x < width && load_y < height) {
                        local_mem[0][local_x + 1] = d_input[load_y * stride + global_x];
                    }
                }
                if (local_y == LOCAL_HEIGHT - 1 || local_y == item.get_local_range(0) - 1) {
                    size_t load_y = (global_y + 1 < height) ? global_y + 1 : global_y;
                    if (global_x < width && global_y < height && load_y < height) {
                        local_mem[local_y + 2][local_x + 1] = d_input[load_y * stride + global_x];
                    }
                }

                if (local_x == 0 && local_y == 0) {
                    size_t load_x = (global_x > 0) ? global_x - 1 : global_x;
                    size_t load_y = (global_y > 0) ? global_y - 1 : global_y;
                    if (load_x < width && load_y < height) {
                        local_mem[0][0] = d_input[load_y * stride + load_x];
                    }
                }
                if ((local_x == LOCAL_WIDTH - 1 || local_x == item.get_local_range(1) - 1) && local_y == 0) {
                    size_t load_x = (global_x + 1 < width) ? global_x + 1 : global_x;
                    size_t load_y = (global_y > 0) ? global_y - 1 : global_y;
                    if (global_x < width && load_x < width && load_y < height) {
                        local_mem[0][local_x + 2] = d_input[load_y * stride + load_x];
                    }
                }
                if (local_x == 0 && (local_y == LOCAL_HEIGHT - 1 || local_y == item.get_local_range(0) - 1)) {
                    size_t load_x = (global_x > 0) ? global_x - 1 : global_x;
                    size_t load_y = (global_y + 1 < height) ? global_y + 1 : global_y;
                    if (global_y < height && load_x < width && load_y < height) {
                        local_mem[local_y + 2][0] = d_input[load_y * stride + load_x];
                    }
                }
                if ((local_x == LOCAL_WIDTH - 1 || local_x == item.get_local_range(1) - 1) &&
                    (local_y == LOCAL_HEIGHT - 1 || local_y == item.get_local_range(0) - 1)) {
                    size_t load_x = (global_x + 1 < width) ? global_x + 1 : global_x;
                    size_t load_y = (global_y + 1 < height) ? global_y + 1 : global_y;
                    if (global_x < width && global_y < height && load_x < width && load_y < height) {
                        local_mem[local_y + 2][local_x + 2] = d_input[load_y * stride + load_x];
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

                if (global_x < width && global_y < height) {
                    uint8_t window[9];

                    size_t idx = 0;
                    for (int dy = 0; dy < 3; ++dy) {
                        for (int dx = 0; dx < 3; ++dx) {
                            window[idx++] = local_mem[local_y + dy][local_x + dx];
                        }
                    }

                    uint8_t median = median_9(window);

                    d_output[global_y * stride + global_x] = median;
                }
            }
        );
    }).wait();

    q.memcpy(output, d_output, output_size * sizeof(uint8_t)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}
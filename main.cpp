#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

#include <sycl/sycl.hpp>
#include "processFileData.h"
#include "medianFilter.h"
#include "medianFilterSIMD.h"
#include "medianFilterGPU.h"
//using namespace sycl;



const size_t data_size = 500000000;//размер массива [500 - 500000000]
const float noise_level = 0.5f;//шум (0.2f / 0)
const float outlier = 1.0f;//выбросы (0.3f / 1.0f)
const size_t outlier_step = 12;//шаг выбросов (0 / 24)

const size_t image_width = 4096;
const size_t image_height = 4096;
const size_t image_stride = image_width;


int main() {
    std::cout << "=== 1D MEDIAN FILTER (Window 1x7) ===" << std::endl;

    auto original_data = generate_test_data(data_size, noise_level, outlier, outlier_step);

    //ОДНОПОТОЧНАЯ ВЕРСИЯ
    std::vector<float> filtered_data(data_size);

    auto start1 = std::chrono::high_resolution_clock::now();
    MedianFilter::median_filter_7(original_data.data(), filtered_data.data(), data_size);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Single thread version: " << duration1.count() << " ms" << std::endl;

    //write_array_to_file("csv/filtered_data.csv", original_data.data(), filtered_data.data(), data_size);


    //GPU ВЕРСИЯ
    std::vector<float> filtered_data_gpu(data_size);

    auto start2 = std::chrono::high_resolution_clock::now();
    MedianFilterGPU::median_filter_7(original_data.data(), filtered_data_gpu.data(), data_size);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "GPU version: " << duration2.count() << " ms" << std::endl;

    //write_array_to_file("csv/filtered_data_gpu.csv", original_data.data(), filtered_data_gpu.data(), data_size);


    //SIMD ВЕРСИЯ
    std::vector<float> filtered_data_simd(data_size);

    auto start3 = std::chrono::high_resolution_clock::now();
    MedianFilterSIMD::median_filter_7(original_data.data(), filtered_data_simd.data(), data_size);
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    std::cout << "SIMD version: " << duration3.count() << " ms" << std::endl;

    //write_array_to_file("csv/filtered_data_simd.csv", original_data.data(), filtered_data_simd.data(), data_size);


    //assert(compare_data(filtered_data.data(), filtered_data_simd.data(), data_size));
    //assert(compare_data(filtered_data.data(), filtered_data_gpu.data(), data_size));
    //std::cout << "Filtered data is equal!" << std::endl;



    std::cout << "\n=== 2D MEDIAN FILTER (Window 3x3) ===" << std::endl;
    std::cout << "Image size: " << image_width << "x" << image_height << " pixels" << std::endl;

    auto original_image = generate_test_image(image_width, image_height, image_stride);

    std::vector<uint8_t> filtered_image(image_height * image_stride);

    auto start2d_1 = std::chrono::high_resolution_clock::now();
    MedianFilter::median_filter_3x3(original_image.data(), filtered_image.data(), image_width, image_height, image_stride);
    auto end2d_1 = std::chrono::high_resolution_clock::now();
    auto duration2d_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end2d_1 - start2d_1);
    std::cout << "Single thread version: " << duration2d_1.count() << " ms" << std::endl;


    std::vector<uint8_t> filtered_image_gpu(image_height * image_stride);

    auto start2d_2 = std::chrono::high_resolution_clock::now();
    MedianFilterGPU::median_filter_3x3(original_image.data(), filtered_image_gpu.data(), image_width, image_height, image_stride);
    auto end2d_2 = std::chrono::high_resolution_clock::now();
    auto duration2d_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2d_2 - start2d_2);
    std::cout << "GPU version: " << duration2d_2.count() << " ms" << std::endl;


    std::vector<uint8_t> filtered_image_simd(image_height * image_stride);

    auto start2d_3 = std::chrono::high_resolution_clock::now();
    MedianFilterSIMD::median_filter_3x3(original_image.data(), filtered_image_simd.data(), image_width, image_height, image_stride);
    auto end2d_3 = std::chrono::high_resolution_clock::now();
    auto duration2d_3 = std::chrono::duration_cast<std::chrono::milliseconds>(end2d_3 - start2d_3);
    std::cout << "SIMD version: " << duration2d_3.count() << " ms" << std::endl;


    return 0;
}
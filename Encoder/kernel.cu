#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <direct.h> 
#include <algorithm>

#include <chrono>

#include "binaryfile.h"

#define BLOCK_SIZE 512
#define USE_STABILIZER true // used just to artistic measures, will slightly hurt performance

class encoder
{
public:
    struct square {
        int x;
        int y;
        int size;
        square() {
            this->x = 0;
            this->y = 0;
            this->size = 0;
        }
        explicit square(int x, int y, int size) {
            this->x = x;
            this->y = y;
            this->size = size;
        }
    };
    encoder(binaryfile& file);
    encoder::square* encodeFrame(int frame, int* squareCount);

private:
    binaryfile& bin;
    int width, height, frames;
    int* d_integral;
    int* d_convolution;
    int* d_indices;
    int* d_sortedIndices;
    int* d_indicesCount;
    encoder::square* d_squares;
    int* d_squaresCount;
};

bool integralImage(int* integralBuffer, const bool* buffer, int width, int height) {
    if (!integralBuffer || !buffer || width <= 0 || height <= 0) {
        return false;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = buffer[y * width + x];

            if (x > 0) {
                sum += integralBuffer[y * width + (x - 1)];
            }
            if (y > 0) {
                sum += integralBuffer[(y - 1) * width + x];
            }
            if (x > 0 && y > 0) {
                sum -= integralBuffer[(y - 1) * width + (x - 1)];
            }

            integralBuffer[y * width + x] = sum;
        }
    }

    return true;
}

// from convolution result
__global__ void reduceIntegralRoi(int* integral, const int width, const int height, const int* indices, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // get position from convolution array
    int indice = indices[0];
    int row = indice / (width - size + 1);
    int col = indice % (width - size + 1);

    int roiHeight = height - row;
    int roiWidth = width - col;

    if (idx >= roiWidth * roiHeight) return;

    int roi_i = idx / roiWidth;
    int roi_j = idx % roiWidth;

    int int_i = roi_i + row;
    int int_j = roi_j + col;

    // multiplication square
    if (roi_i < size && roi_j < size) {
        integral[int_i * width + int_j] -= (roi_i + 1) * (roi_j + 1);
    }

    // extend right
    else if (roi_i < size && roi_j < roiWidth) {
        integral[int_i * width + int_j] -= (roi_i + 1) * size;
    }

    // extend down
    else if (roi_i < roiHeight && roi_j < size) {
        integral[int_i * width + int_j] -= size * (roi_j + 1);
    }

    // extend beyond
    else if (roi_i < roiHeight && roi_j < roiWidth) {
        integral[int_i * width + int_j] -= size * size;
    }
}

__global__ void convolveIntegral(int* buffer, const int* integral, const int width, const int height, const int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int totalElements = (width - size + 1) * (height - size + 1);
    if (idx >= totalElements) return;

    int i = idx / (width - size + 1);
    int j = idx % (width - size + 1);

    // guarantee that it wont go out of bound both right and down
    int val = integral[(i + size - 1) * width + (j + size - 1)];
    if (i > 0) val -= integral[(i - 1) * width + (j + size - 1)];
    if (j > 0) val -= integral[(i + size - 1) * width + (j - 1)];
    if (i > 0 && j > 0) val += integral[(i - 1) * width + (j - 1)];

    buffer[i * (width - size + 1) + j] = val;
}

__global__ void findIndices(const int* array, int* indices, int* count, const int target, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (array[idx] == target) {
        int pos = atomicAdd(count, 1);
        indices[pos] = idx;
    }
}

// from convolution result
__global__ void addSquare(encoder::square* squares, int* squaresCount, const int* indices, const int width, const int height, const int size, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;
    int indice = indices[idx];
    int row = indice / (width - size + 1);
    int col = indice % (width - size + 1);

    int pos = atomicAdd(squaresCount, 1);
    squares[pos].x = col;
    squares[pos].y = row;
    squares[pos].size = size;
}

void getLaunchConfig(const int n, int* blockSize, int* gridSize) {
    *blockSize = BLOCK_SIZE;
    *gridSize = (n + *blockSize - 1) / *blockSize;
}

encoder::encoder(binaryfile& bin) : bin(bin) {
    width = bin.getMetadata<int>(binaryfile::metadatatype::Width);
    height = bin.getMetadata<int>(binaryfile::metadatatype::Height);
    frames = bin.getMetadata<int>(binaryfile::metadatatype::Frames);

    cudaMalloc(&d_integral, width * height * sizeof(int));
    cudaMalloc(&d_convolution, width * height * sizeof(int));
    cudaMalloc(&d_indices, width * height * sizeof(int));
    cudaMalloc(&d_sortedIndices, width * height * sizeof(int));
    cudaMalloc(&d_indicesCount, sizeof(int));
    cudaMalloc((void**)&d_squares, width * height * sizeof(encoder::square));
    cudaMalloc(&d_squaresCount, sizeof(int));

    cudaMemset(d_indicesCount, 0, sizeof(int));
}

encoder::square* encoder::encodeFrame(int frame, int* squareCount) {
    bool* h_buffer = new bool[width * height];
    bin.loadFrame(h_buffer, width, height, frame);

    int* h_integral = new int[width * height];
    integralImage(h_integral, h_buffer, width, height);
    cudaMemcpy(d_integral, h_integral, width * height * sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_buffer;
    delete[] h_integral;

    int h_indicesCount = 0;

    int kernelSize = min(width, height);
    int blockSize, gridSize;

    cudaMemset(d_squaresCount, 0, sizeof(int));

    while (kernelSize > 0) {
        int convolutionSize = (width - kernelSize + 1) * (height - kernelSize + 1);

        getLaunchConfig(convolutionSize, &blockSize, &gridSize);
        convolveIntegral << <gridSize, blockSize >> > (d_convolution, d_integral, width, height, kernelSize);
        cudaDeviceSynchronize();

        getLaunchConfig(convolutionSize, &blockSize, &gridSize);
        cudaMemset(d_indicesCount, 0, sizeof(int));
        findIndices << <gridSize, blockSize >> > (d_convolution, d_indices, d_indicesCount, kernelSize * kernelSize, convolutionSize);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_indicesCount, d_indicesCount, sizeof(int), cudaMemcpyDeviceToHost);

        // trivial condition: add all indices if kernel size is 1
        if (kernelSize == 1) {
            getLaunchConfig(h_indicesCount, &blockSize, &gridSize);
            addSquare << <gridSize, blockSize >> > (d_squares, d_squaresCount, d_indices, width, height, kernelSize, h_indicesCount);
            cudaDeviceSynchronize();
            break;
        }

        // sort the indices to stabilize
        if (USE_STABILIZER) {
            int* h_indices = new int[h_indicesCount];
            cudaMemcpy(h_indices, d_indices, h_indicesCount * sizeof(int), cudaMemcpyDeviceToHost);

            std::sort(h_indices, h_indices + h_indicesCount);

            cudaMemcpy(d_indices, h_indices, h_indicesCount * sizeof(int), cudaMemcpyHostToDevice);
        }
        
        if (h_indicesCount > 0) {
            addSquare << <1, 1 >> > (d_squares, d_squaresCount, d_indices, width, height, kernelSize, 1);
            cudaDeviceSynchronize();

            getLaunchConfig(width * height, &blockSize, &gridSize);
            reduceIntegralRoi << <gridSize, blockSize >> > (d_integral, width, height, d_indices, kernelSize);
            cudaDeviceSynchronize();

            continue;
        }

        kernelSize -= 1;
    }

    cudaMemcpy(squareCount, d_squaresCount, sizeof(int), cudaMemcpyDeviceToHost);

    encoder::square* h_squares = new encoder::square[*squareCount];
    cudaMemcpy(h_squares, d_squares, *squareCount * sizeof(encoder::square), cudaMemcpyDeviceToHost);

    return h_squares;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filepath> <outputpath>\n";
        return 1;
    }

    std::string filePath = argv[1];
    std::string ouputPath = argv[2];
    std::cout << "File path: " << filePath << std::endl;
    std::cout << "Output path: " << ouputPath << std::endl;

    binaryfile bin(filePath);

    int width = bin.getMetadata<int>(binaryfile::metadatatype::Width);
    int height = bin.getMetadata<int>(binaryfile::metadatatype::Height);
    int frameCount = bin.getMetadata<int>(binaryfile::metadatatype::Frames);
    float fps = bin.getMetadata<float>(binaryfile::metadatatype::Fps);

    std::cout << "Width: " << width << ", Height: " << height << ", FPS: " << fps << ", Frames: " << frameCount << std::endl;

    if (height > 1024) {
        fprintf(stderr, "Height more than 1024 pixels is not yet supported.");
        return 1;
    }

    std::ofstream outputFile(ouputPath, std::ios::out | std::ios::binary);
    if (!outputFile) {
        std::cerr << "Error: Could not create or open file!" << std::endl;
        return 1;
    }
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }

    int* squaresPerFrame = new int[frameCount];
    std::vector<encoder::square> squares = std::vector<encoder::square>();

    std::cout << "Loading data..." << std::endl;
    auto loadStart = std::chrono::high_resolution_clock::now();
    encoder encoder = encoder::encoder(bin);
    auto loadEnd = std::chrono::high_resolution_clock::now();
    std::cout << "Data load took " << std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadStart).count() << "ms." << std::endl;

    auto encodeStart = std::chrono::high_resolution_clock::now();
    for (int frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        std::cout << "\rEncoding frame " << frameIndex + 1 << " out of " << frameCount << "...";
        int squareCount = 0;
        encoder::square* frameSquares = encoder.encodeFrame(frameIndex, &squareCount);

        squaresPerFrame[frameIndex] = squareCount;
        squares.insert(squares.end(), frameSquares, frameSquares + squareCount);

        delete[] frameSquares;
    }
    auto encodeEnd = std::chrono::high_resolution_clock::now();

    std::cout << std::endl << "Encoder took " << std::chrono::duration_cast<std::chrono::milliseconds>(encodeEnd - encodeStart).count() << "ms and resulted in " << squares.size() << " squares!" << std::endl;
    
    std::cout << "Writing to file..." << std::endl;

    auto writeStart = std::chrono::high_resolution_clock::now();
    int startingIndex = 0;
    for (int frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        uint32_t frameSquareCount = static_cast<uint32_t>(squaresPerFrame[frameIndex]);
        outputFile.write(reinterpret_cast<char*>(&frameSquareCount), sizeof(uint32_t));
        for (int squareIndex = 0; squareIndex < squaresPerFrame[frameIndex]; squareIndex++) {
            encoder::square square = squares[startingIndex + squareIndex];

            uint16_t x = static_cast<uint16_t>(square.x);
            uint16_t y = static_cast<uint16_t>(square.y);
            uint16_t size = static_cast<uint16_t>(square.size);

            outputFile.write(reinterpret_cast<char*>(&y), sizeof(uint16_t));
            outputFile.write(reinterpret_cast<char*>(&x), sizeof(uint16_t));
            outputFile.write(reinterpret_cast<char*>(&size), sizeof(uint16_t));
        }
        startingIndex += squaresPerFrame[frameIndex];
    }
    auto writeEnd = std::chrono::high_resolution_clock::now();

    std::cout << "Writing to file took " << std::chrono::duration_cast<std::chrono::milliseconds>(writeEnd - writeStart).count() << "ms." << std::endl;

    return 0;
}
#include "binaryfile.h"

#include <cmath>
#include <stdexcept>
#include <vector>

static bool getFileMetadata(std::ifstream& stream, int* width, int* height, float* fps) {
    stream.seekg(0, stream.beg);

    stream.read(reinterpret_cast<char*>(width), sizeof(int));
    stream.read(reinterpret_cast<char*>(height), sizeof(int));
    stream.read(reinterpret_cast<char*>(fps), sizeof(float));

    return true;
}

static bool getRangeBitWise(std::ifstream& stream, bool* buffer, const int startIndex, const int size) {
    if (!stream.is_open()) {
        std::cerr << "Error: Stream is not open." << std::endl;
        return false;
    }

    if (startIndex < 0 || size < 0) {
        std::cerr << "Error: Invalid startIndex or size." << std::endl;
        return false;
    }

    int startByte = startIndex / 8;
    int startBit = startIndex % 8;

    int bytesToRead = std::ceil((double)(startIndex + size) / 8.0) - startByte;

    if (bytesToRead <= 0) {
        std::cerr << "Error: No bytes to read. Check startIndex and size." << std::endl;
        return false;
    }

    std::vector<unsigned char> byteBuffer(bytesToRead);

    stream.seekg(startByte);

    if (!stream.read(reinterpret_cast<char*>(byteBuffer.data()), bytesToRead)) {
        std::cerr << "Error: Failed to read from stream. Possibly reached end of file." << std::endl;
        return false;
    }

    for (int i = 0; i < size; ++i) {
        int byteIndex = (startIndex + i) / 8 - startByte;
        int bitIndex = (startIndex + i) % 8;

        if (byteIndex >= bytesToRead) {
            std::cerr << "Error: Index out of range. Check startIndex and size." << std::endl;
            return false;
        }

        buffer[i] = (byteBuffer[byteIndex] >> (7 - bitIndex)) & 1;
    }

    return true;
}

static bool getFrameCount(std::ifstream& stream, int* frameCount) {

    int width, height;
    float _;

    if (!getFileMetadata(stream, &width, &height, &_)) {
        return false;
    }

    // calculate frame size from file size
    stream.seekg(0, stream.end);
    int fileSize = stream.tellg();

    if (fileSize <= 0) {
        std::cerr << "Error: Invalid file size.\n";
        return false;
    }

    size_t fileSizeBits = (fileSize - 12) * 8;
    *frameCount = fileSizeBits / (width * height);

    return true;
}

binaryfile::binaryfile(std::string& filePath) {
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filePath);
    }

    if (!loadMetadata()) {
        throw std::runtime_error("Error loading metadata: " + filePath);
    }
}

binaryfile::~binaryfile() {
    if (file.is_open()) {
        file.close();
    }
}

bool binaryfile::is_open() const {
    return file.is_open();
}

bool binaryfile::loadMetadata() {
    if (!file.is_open()) {
        return false;
    }

    if (!getFileMetadata(file, &width, &height, &fps)) {
        return false;
    }

    if (!getFrameCount(file, &frames)) {
        return false;
    }

    return true;
}

bool binaryfile::loadFrame(bool* buffer, int width, int height, int frame) {
    const size_t metadataSize = 96;
    const size_t frameSize = width * height;
    const size_t frameOffset = frameSize * frame;

    getRangeBitWise(file, buffer, frameOffset + metadataSize, frameSize);

    return true;
}


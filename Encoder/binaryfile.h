#pragma once

#include <iostream>
#include <fstream>

class binaryfile {
private:
	std::ifstream file;
	std::string filePath;

	int width, height, frames;
	float fps;

    bool loadMetadata();

public:
    enum metadatatype {
        Width, Height, Frames, Fps
    };

	binaryfile(std::string& filePath);

	~binaryfile();

	bool is_open() const;

	bool loadFrame(bool* buffer, int width, int height, int frame);

    template <typename T>
    T getMetadata(metadatatype type) const {
        switch (type) {
        case metadatatype::Width:
            return static_cast<T>(width);
        case metadatatype::Height:
            return static_cast<T>(height);
        case metadatatype::Frames:
            return static_cast<T>(frames);
        case metadatatype::Fps:
            return static_cast<T>(fps);
        default:
            throw std::runtime_error("Invalid metadata type.");
        }
    }
};


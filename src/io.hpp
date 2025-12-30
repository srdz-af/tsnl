#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct MMapArrayBase {
    void* data = nullptr;
    size_t bytes = 0;
    int fd = -1;
};

template <typename T>
struct MMapArray {
    const T* data = nullptr;
    size_t size = 0;
    MMapArrayBase base;

    const T& operator[](size_t i) const { return data[i]; }
    bool empty() const { return size == 0; }
};

bool map_readonly(const std::string& path, MMapArrayBase& out);
void unmap(MMapArrayBase& arr);
size_t file_size(const std::string& path);
bool file_exists(const std::string& path);

template <typename T>
bool map_array(const std::string& path, MMapArray<T>& out) {
    if (!map_readonly(path, out.base)) return false;
    if (out.base.bytes % sizeof(T) != 0) {
        unmap(out.base);
        return false;
    }
    out.size = out.base.bytes / sizeof(T);
    out.data = reinterpret_cast<const T*>(out.base.data);
    return true;
}

template <typename T>
bool write_array(const std::string& path, const std::vector<T>& data);

struct Triple {
    uint32_t h;
    uint32_t r;
    uint32_t t;
};

bool map_triples(const std::string& path, MMapArray<Triple>& out);
std::vector<std::string> split_paths(const std::string& paths, char sep = ',');


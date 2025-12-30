#include "io.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <sstream>

bool map_readonly(const std::string& path, MMapArrayBase& out) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open " << path << ": " << strerror(errno) << "\n";
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        std::cerr << "Failed to stat " << path << ": " << strerror(errno) << "\n";
        close(fd);
        return false;
    }
    size_t bytes = static_cast<size_t>(st.st_size);
    void* ptr = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "mmap failed for " << path << ": " << strerror(errno) << "\n";
        close(fd);
        return false;
    }
    out.data = ptr;
    out.bytes = bytes;
    out.fd = fd;
    return true;
}

void unmap(MMapArrayBase& arr) {
    if (arr.data && arr.data != MAP_FAILED) {
        munmap(arr.data, arr.bytes);
    }
    if (arr.fd >= 0) close(arr.fd);
    arr.data = nullptr;
    arr.bytes = 0;
    arr.fd = -1;
}

size_t file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
    return static_cast<size_t>(st.st_size);
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

template <typename T>
bool write_array(const std::string& path, const std::vector<T>& data) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        std::cerr << "Failed to open " << path << " for write: " << strerror(errno) << "\n";
        return false;
    }
    size_t bytes = data.size() * sizeof(T);
    ssize_t wrote = write(fd, reinterpret_cast<const char*>(data.data()), bytes);
    close(fd);
    return wrote == static_cast<ssize_t>(bytes);
}

template bool write_array<uint32_t>(const std::string&, const std::vector<uint32_t>&);
template bool write_array<uint16_t>(const std::string&, const std::vector<uint16_t>&);

bool map_triples(const std::string& path, MMapArray<Triple>& out) {
    if (!map_array(path, out)) return false;
    return true;
}

std::vector<std::string> split_paths(const std::string& paths, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(paths);
    std::string token;
    while (std::getline(ss, token, sep)) {
        if (!token.empty()) out.push_back(token);
    }
    return out;
}

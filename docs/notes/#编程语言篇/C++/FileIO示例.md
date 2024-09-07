### File IO Functions

```c++
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <climits>
#include <sys/stat.h>

class FileIO {
public:
    static bool isFileExist(const std::string &filename) {
        std::ifstream file(filename);
        return file.good();
    }

    static bool deleteFile(const std::string &filename) {
        if (!isFileExist(filename)) {
            LOG(ERROR) << "delete file : " << filename << " does not exist";
            return false;
        }
        if (std::remove(filename.c_str()) != 0) {
            LOG(ERROR) << "delete file : " << filename << " failed";
            return false;
        }
        return true;
    }

    static bool writeToFile(const std::string &filename, const std::string &data) {
        if (isFileExist(filename)) {
            LOG(ERROR) << "write file : " << filename << " already exist";
            return false;
        }
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (file) {
            file.write(data.c_str(), data.size());
            file.close();
            return true;
        }
        return false;
    }

    /*
     * replace ~ to home dir path
     * */
    static bool getFilePath(const std::string &filename, std::string &filePath) {
        const char *homePath = getenv("HOME");
        if (homePath == nullptr) {
            std::cerr << "Unable to retrieve HOME environment variable" << std::endl;
            return false;
        }

        filePath = filename;
        if (filename[0] == '~') {
            filePath.replace(0, 1, homePath);
        }
        return true;
    }


    static bool createDirectory(const std::string &path_str) {
        struct stat info{};

        auto path = path_str.c_str();
        if (stat(path, &info) != 0) {  // 检查文件夹是否已经存在
            if (mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0) {
                LOG(INFO) << "文件夹创建成功！";
                return true;
            } else {
                LOG(INFO) << "文件夹创建失败！";
                return false;
            }
        } else {
            LOG(INFO) << "文件夹已经存在，无需创建！";
            return true;
        }
    }

    static bool writeToFile(const std::string &filename, const std::vector<unsigned char> &data) {
        if (isFileExist(filename)) {
            LOG(ERROR) << "write file : " << filename << " already exist";
            return false;
        }
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (file) {
            file.write(reinterpret_cast<const char *>(data.data()), data.size());
            file.close();
            return true;
        }
        return false;
    }

    static bool readFromFile(const std::string &filename, std::string &outData) {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (file) {
            outData.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            file.close();
            return true;
        }
        return false;
    }

    static bool readBytesFromFile(const std::string &filename, std::vector<unsigned char> &outData) {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (file) {
            outData.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            file.close();
            return true;
        }
        return false;
    }
};
```
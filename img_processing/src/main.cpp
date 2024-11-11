#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"
#include "manager.h"
#include "jsonUtils.h"
#include "log_manager.h"
#include "distance.h"

using json = nlohmann::json;
using namespace std;
using namespace cv;

void runOnVideo(Manager &manager, string videoPath, bool isTravel)
{
    // Convert Windows file path to WSL file path format
    if (videoPath.length() >= 3 && videoPath[1] == ':') {
        // Convert to lowercase
        char driveLetter = tolower(static_cast<unsigned char>(videoPath[0]));
        videoPath = "/mnt/" + string(1, driveLetter) + videoPath.substr(2);
        // Replace backslashes with forward slashes
        replace(videoPath.begin(), videoPath.end(), '\\', '/');
    }
    // open the video
    VideoCapture capture(videoPath);
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    if (!capture.isOpened()) {
        LogManager::logErrorMessage(ErrorType::VIDEO_ERROR, "video not found");
        throw runtime_error("video not found");
        return;
    }
    // run on video
    while (1) {
        capture >> frame;
        if (frame.empty()) {
            LogManager::logInfoMessage(InfoType::MEDIA_FINISH);
            break;
        }
        int result = manager.processing(frame, isTravel);
        if (result == -1)
            return;
    }
}

void mainDemo(Manager &manager)
{
    string path;
    int focalLength;
    bool isTravel;
    // Open the JSON file
    std::ifstream file("../data.json");

    if (!file.is_open()) {
        LogManager::logErrorMessage(ErrorType::FILE_ERROR,
                                    "Failed to open the file");
        throw runtime_error("Failed to open the file");
        return;
    }

    // Read the content of the file into a JSON object
    json jsonData;
    file >> jsonData;

    // Check if the JSON data is an array
    if (jsonData.is_array()) {
        // Iterate over each object in the array
        for (const auto &obj : jsonData) {
            manager.setIterationCnt(1);
            if (obj.find("path") != obj.end() && obj["path"].is_string()) {
                path = obj["path"];
            }

            if (obj.find("focal_length") != obj.end() &&
                obj["focal_length"].is_number_integer()) {
                focalLength = obj["focal_length"];
            }

            if (obj.find("is_travel") != obj.end() &&
                obj["is_travel"].is_boolean()) {
                isTravel = obj["is_travel"];
            }
            Distance *distance = manager.getDistance();
            distance->setFocalLength(focalLength);
            runOnVideo(manager, path, isTravel);
        }
    }
}

int main()
{
    int processID = readFromJson("ID");
    Manager manager(processID);
    manager.init();
    mainDemo(manager);
    return 0;
}
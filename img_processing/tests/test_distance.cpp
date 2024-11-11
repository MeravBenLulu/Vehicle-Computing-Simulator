#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "detector.h"
#include "distance.h"
#include "log_manager.h"
#include "manager.h"

using namespace std;
using namespace cv;

TEST(DistanceTest, DistanceWithCalibration)
{
    // Load a real image from file
    string imagePath = "../tests/images/black_line.JPG";
    Mat calibrationImage;
    calibrationImage = imread(imagePath);
    if (calibrationImage.empty()) {
        LogManager::logErrorMessage(ErrorType::IMAGE_ERROR,
                                    "Could not load image");
        throw runtime_error("Could not open or find the image");
    }

    Distance distance;
    distance.setFocalLength(calibrationImage);
    // Load a real image from file
    string imagePath2 = "../tests/images/parking_car.JPG";
    Mat carImage;
    carImage = imread(imagePath2);
    if (carImage.empty()) {
        LogManager::logErrorMessage(ErrorType::IMAGE_ERROR,
                                    "Could not load image");
        throw runtime_error("Could not open or find the image");
    }

    // Wrap it in a shared_ptr
    shared_ptr<Mat> frame = make_shared<Mat>(carImage);
    vector<ObjectInformation> output;
    // Create Detector instance
    Detector detector;

    // bool is_cuda = false; // or true, depending on your setup
    detector.init(false);

    // Perform detection
    detector.detect(frame, false, output);

    // Check if output is not empty
    ASSERT_FALSE(output.empty());

    distance.findDistance(output);

    bool isFind = false;
    // Going through all the detected objects and sending them for a distance test
    for (auto detectionObject : output) {
        LogManager::logInfoMessage(InfoType::DISTANCE,
                                   to_string(detectionObject.distance));
        if (detectionObject.distance > 3 && detectionObject.distance < 4)
            isFind = true;
    }
    ASSERT_TRUE(isFind);
}

#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "detector.h"
#include "distance.h"
#include "dynamic_tracker.h"
#include "velocity.h"

using namespace std;
using namespace cv;

TEST(TVelocity, calculate_TVelocity)
{
    // Load a real image from file
    string imagePath = "../tests/images/black_line.JPG";
    Mat calibrationImage;
    calibrationImage = imread(imagePath);
    if (calibrationImage.empty()) {
        throw runtime_error("Could not open or find the image");
    }
    Distance distance;
    distance.setFocalLength(calibrationImage);
    Detector detector;
    DynamicTracker tracker;
    Velocity velocity;
    velocity.init(0.04);
    detector.init(false);
    tracker.init();
    VideoCapture capture("../tests/images/one_car.mp4");
    Mat frame;
    capture.read(frame);
    while (!frame.empty()) {
        shared_ptr<Mat> f1 = make_shared<Mat>(frame);

        // reset objects
        auto output = vector<ObjectInformation>();
        detector.detect(f1, true, output);
        tracker.startTracking(f1, output);

        // after reset we not calc velocity
        capture.read(frame);
        if (frame.empty())
            return;
        shared_ptr<Mat> frame1 = make_shared<Mat>(frame);
        tracker.tracking(frame1, output);
        distance.findDistance(output);

        for (int i = 0; i < 9; i++) {
            capture.read(frame);
            if (frame.empty())
                return;
            shared_ptr<Mat> frame1 = make_shared<Mat>(frame);
            tracker.tracking(frame1, output);
            distance.findDistance(output);
            velocity.calculateVelocities(output);
            if (output.size() == 1)
                LogManager::logDebugMessage(
                    DebugType::PRINT,
                    std::to_string(output[0].velocity.value()));
        }
        capture.read(frame);
    }
}
#include <iostream>
#include <opencv2/highgui.hpp>
#include <gtest/gtest.h>
#include "lane_detector.h"
#include "log_manager.h"

using namespace std;
using namespace cv;

TEST(TLaneDetector, CheckRun)
{
    cout << "start" << endl;
    string path = "/home/tirtza/train6/171.mov";
    VideoCapture cap(path);

    if (!cap.isOpened()) {
        LogManager::logErrorMessage(ErrorType::VIDEO_ERROR,
                                    "Could not load video");

        throw runtime_error("Could not load video");
    }

    LaneDetector laneDetector;

    laneDetector.init();

    while (true) {
        Mat frame;
        cap >> frame;

        auto sharedFrame = make_shared<Mat>(frame);

        laneDetector.manageLaneDetector(sharedFrame);
        laneDetector.drawLane(sharedFrame);

        imshow("lane-detection", *sharedFrame);
        if (waitKey() == 27)
            break;
    }

    cap.release();
}
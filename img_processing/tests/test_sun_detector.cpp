#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "sun_detector.h"

using namespace std;
using namespace cv;

TEST(SunDetectorTest, detectSun)
{
    VideoCapture capture("../tests/images/sun_in_street.mov");

    if (!capture.isOpened()) {
        // LogManager::logErrorMessage(ErrorType::VIDEO_ERROR, "video not found");
        // throw runtime_error("video not found");
        cout << "couldn't open video" << endl;
        return;
    }
    Mat img;
    SunDetector sd;
    while (1) {
        capture >> img;

        if (img.empty()) {
            // LogManager::logInfoMessage(InfoType::MEDIA_FINISH);
            cout << "media finish" << endl;
            break;
        }
        shared_ptr<Mat> frame = make_shared<Mat>(img);
        sd.detectSun(frame);
        sd.drowSun(frame);
        imshow("sun", *frame);
        int key = waitKey(1);
        if (key == 27) {
            break;
        }
    }
}
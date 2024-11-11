#ifndef __SUN_DETECTOR_H__
#define __SUN_DETECTOR_H__

#include <opencv2/opencv.hpp>

class SunDetector {
   public:
    void detectSun(const std::shared_ptr<cv::Mat> &frame);
    void drowSun(std::shared_ptr<cv::Mat> &frame);

   private:
    cv::Point2f center;
    float radius;
    bool isSun;
};

#endif  // __SUN_DETECTOR_H__
#ifndef __DISTANCE_H__
#define __DISTANCE_H__
#define PERSON_HEIGHT 1700
#define CAR_WIDTH 2000

#include "log_manager.h"
#include "object_information_struct.h"

class Distance {
   public:
    void findDistance(std::vector<ObjectInformation> &objectInformation);
    void setFocalLength(const cv::Mat &image);
    void setFocalLength(double focalLength);
    void drawDistance(const std::shared_ptr<cv::Mat> image,
                      const std::vector<ObjectInformation> &objects) const;

   private:
    double focalLength;
    void findFocalLength(const cv::Mat &image);
    void addDistance(float distance, ObjectInformation &obj);
};

#endif  //__DISTANCE_H__
#ifndef __VELOCITY_H__
#define __VELOCITY_H__

#include <opencv2/opencv.hpp>
#include <numeric>
#include "object_information_struct.h"

class Velocity {
   public:
    Velocity() {}
    void calculateVelocities(std::vector<ObjectInformation> &objects);
    void init(double frameTimeDiff);
    void drawVelocity(const std::shared_ptr<cv::Mat> image,
                      const std::vector<ObjectInformation> &objects) const;

   private:
    double frameTimeDiff;
    const float alpha = 0.17f;
    void calculateVelocity(ObjectInformation &object);
    float averageDistanceChange(const ObjectInformation &obj) const;
    void smoothAndUpdateVelocity(float newVelocity, ObjectInformation &obj);
};
#endif  //__VELOCITY_H__
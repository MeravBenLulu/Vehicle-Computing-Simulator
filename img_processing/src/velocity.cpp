#include "velocity.h"

using namespace std;
using namespace cv;

void Velocity::init(double frameTimeDiff)
{
    this->frameTimeDiff = frameTimeDiff;
}
void Velocity::calculateVelocities(vector<ObjectInformation> &objects)
{
    for (auto &object : objects) {
        calculateVelocity(object);
    }
}
void Velocity::calculateVelocity(ObjectInformation &object)
{
    float distanceAvg = averageDistanceChange(object);
    if (distanceAvg != -1) {
        float velocity = distanceAvg / this->frameTimeDiff;
        smoothAndUpdateVelocity(velocity, object);
    }
}

float Velocity::averageDistanceChange(const ObjectInformation &obj) const
{
    if (obj.prevDistances.size() < 2)
        return -1;
    float totalChange = 0.0;
    for (size_t i = 1; i < obj.prevDistances.size(); ++i) {
        totalChange += (obj.prevDistances[i] - obj.prevDistances[i - 1]);
    }
    return totalChange / (obj.prevDistances.size() - 1);
}

void Velocity::smoothAndUpdateVelocity(float newVelocity,
                                       ObjectInformation &obj)
{
    if (obj.velocity.has_value()) {
        obj.velocity =
            (alpha * newVelocity) + ((1 - alpha) * obj.velocity.value());
    }
    else
        obj.velocity = newVelocity;
}

void Velocity::drawVelocity(const std::shared_ptr<Mat> image,
                            const std::vector<ObjectInformation> &objects) const
{
    // Font properties
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.6;
    int thickness = 2;
    int baseline = 0;

    // Calculate text sizes
    Size velocityTextSize =
        getTextSize("velocity", fontFace, fontScale, thickness, &baseline);

    for (auto &obj : objects) {
        // Check if velocity has a value
        if (obj.velocity.has_value()) {
            std::stringstream ssVelocity;
            ssVelocity << std::fixed << std::setprecision(2)
                       << obj.velocity.value();

            Point velocityTextOrg(obj.position.x + 5, obj.position.y - 7);

            // Draw outline for velocity text
            putText(*image, ssVelocity.str(), velocityTextOrg, fontFace,
                    fontScale, Scalar(0, 0, 0), thickness + 3);
            // Write the velocity text
            putText(*image, ssVelocity.str(), velocityTextOrg, fontFace,
                    fontScale, Scalar(255, 255, 0), thickness);
        }
    }
}

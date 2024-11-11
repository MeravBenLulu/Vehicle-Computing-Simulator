#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "dynamic_tracker.h"
#include "manager.h"

using namespace std;
using namespace cv;

void DynamicTracker::init() {}

void DynamicTracker::startTracking(
    const shared_ptr<Mat> &frame,
    const vector<ObjectInformation> &detectionOutput)
{
    trackers.clear();
    failedCount.clear();
    this->frame = frame;
    // Create trackers for each detected object
    for (const auto &objectInformation : detectionOutput) {
        Ptr<Tracker> tracker = TrackerCSRT::create();
        tracker->init(*frame, objectInformation.position);
        trackers.push_back(tracker);
        failedCount.push_back(0);
    }
}

void DynamicTracker::tracking(const shared_ptr<Mat> &frame,
                              std::vector<ObjectInformation> &objectInformation)
{
    this->frame = frame;
    Rect bbox;
    std::vector<size_t> toRemove;  // Store indices to remove

    // Update tracking results for each tracker
    for (size_t i = 0; i < trackers.size(); ++i) {
        bool ok = trackers[i]->update(*frame, bbox);
        if (ok) {
            objectInformation[i].prevPosition = objectInformation[i].position;
            objectInformation[i].position = bbox;
            failedCount[i] = 0;  // Reset failure count on successful tracking
        }
        else {
            failedCount[i]++;
            if (failedCount[i] > maxFailures) {
                toRemove.push_back(i);  // Mark for removal
            }
        }
    }

    // Remove trackers, objectInformation, and failedCount entries after the loop
    for (int i = toRemove.size() - 1; i >= 0; --i) {
        size_t idx = toRemove[i];
        trackers.erase(trackers.begin() + idx);
        objectInformation.erase(objectInformation.begin() + idx);
        failedCount.erase(failedCount.begin() + idx);
    }
}

void DynamicTracker::drawTracking(
    const shared_ptr<Mat> image, const vector<ObjectInformation> &objects) const
{
    for (const auto &objectInformation : objects) {
        Scalar boxColor =
            (objectInformation.distance < (Alerter::MIN_LEGAL_DISTANCE))
                ? Scalar(0, 0, 255)
                : Scalar(0, 255, 0);
        rectangle(*image, objectInformation.position, boxColor, 2);
    }
}

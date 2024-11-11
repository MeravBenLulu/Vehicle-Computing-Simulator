#ifndef __MANAGER_H__
#define __MANAGER_H__

#include <opencv2/opencv.hpp>
#include "alerter.h"
#include "communication.h"
#include "detector.h"
#include "distance.h"
#include "dynamic_tracker.h"
#include "lane_detector.h"
#include "log_manager.h"
#include "sun_detector.h"
#include "velocity.h"

class Manager {
   public:
    static logger imgLogger;
    Manager(int processID);
    // Gets the currentFrame and sends it for detection and then tracking,
    // finally if necessary sends a alert
    int processing(const cv::Mat &newFrame, bool mode);
    // init all variabels and creat the instance of distance
    void init();
    void setIterationCnt(int cnt);
    Distance *getDistance();

   private:
    Communication communication;
    std::shared_ptr<cv::Mat> prevFrame;
    std::shared_ptr<cv::Mat> currentFrame;
    std::vector<ObjectInformation> prevOutput;
    std::vector<ObjectInformation> currentOutput;
    Detector detector;
    Velocity velocity;
    DynamicTracker dynamicTracker;
    Distance distance;
    Alerter alerter;
    SunDetector sunDetector;
    LaneDetector laneDetector;
    int longTime;
    int iterationCnt;
    uint32_t destID;
    uint32_t processID;
    bool isTravel;

    // Moves the current image to the prevFrame
    // and clears the memory of the currentFrame;
    void prepareForTheNext();
    int drawOutput();
    bool isDetect();
    bool isResetTracker();
    bool isTrack();
    bool isCalcVelocity();
    void sendAlerts(std::vector<std::vector<uint8_t>> &alerts);
    int readIdFromJson(const char *target);
};
#endif  //__MANAGER_H__

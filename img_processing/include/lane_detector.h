
#ifndef __LANE_DETECTION__
#define __LANE_DETECTION__

#include <cmath>
#include <opencv2/highgui.hpp>
#include <stdexcept>
#include "regression.h"
#include "log_manager.h"

#define _USE_MATH_DEFINES
#define CAR_IN_IMAGE 80

struct LaneDrawingInfo {
    int rightX1;  //x bottom right
    int rightX2;  //x top right
    int leftX1;   //x bottom left
    int leftX2;   //x top left
    int y1;       //y bottom right & left
    int y2;       //y top right & left
};

class LaneDetector {
   public:
    void init();
    void manageLaneDetector(std::shared_ptr<cv::Mat> frame);
    void drawLane(std::shared_ptr<cv::Mat> img);

   private:
    std::shared_ptr<cv::Mat> image;
    bool first;
    bool withoutCar;
    int imgCols;
    int imgRows;
    LaneDrawingInfo drawingInfo;

    /**
    * Returns true when the image is classified as daytime. 
    * Note: this is based on the mean pixel value of an image and might not
    * always lead to accurate results.
    */
    bool isDayTime();

    /**
    * Filter source image so that only the white and yellow pixels remain.
    * A gray filter will also be added if the source image is classified as taken during the night.
    * One assumption for lane detection here is that lanes are either white or yellow.
    * @param isDayTime true if image is taken during the day, false if at night
    * @return Mat filtered image
    */
    cv::Mat filterColors(bool isDayTime);

    /**
    * Apply grayscale transform on image.
    * @return grayscale image
    */
    cv::Mat applyGrayscale(cv::Mat source);

    /**
    * Apply Gaussian blur to image.
    * @return blurred image
    */
    cv::Mat applyGaussianBlur(cv::Mat source);

    /**
    * Detect edges of image by applying canny edge detection.
    * @return image with detected edges
    */
    cv::Mat applyCanny(cv::Mat source);

    /**
    * Apply an image mask. 
    * Only keep the part of the image defined by the
    * polygon formed from four points. The rest of the image is set to black.
    * @return Mat image with mask
    */
    cv::Mat RegionOfInterest(cv::Mat source);

    /**
    * Returns a vector with the detected hough lines.
    * @param canny image resulted from a canny transform
    * @param source image on which hough lines are drawn
    * @param drawHough draw detected lines on source image if true. 
    * It will also show the image with de lines drawn on it, which is why
    * it is not recommended to pass in true when working with a video. 
    * @return vector<Vec4i> contains hough lines.
    */
    std::vector<cv::Vec4i> houghLines(cv::Mat canny, cv::Mat source,
                                      bool drawHough);

    /**
    * Creates mask and blends it with source image so that the lanes
    * are drawn on the source image.
    * @param lines vector < vec4i > holding the lines
    * @return Mat image with lines drawn on it
    */
    bool drawLanes(std::vector<cv::Vec4i> lines);

    /**
    * Drawing the lane on the road only
    */
    void cutCar();
};

#endif /*__LANE_DETECTION__*/

#include "lane_detector.h"

using namespace std;
using namespace cv;

void LaneDetector::init()
{
    this->first = true;
    this->image = std::make_shared<cv::Mat>();
    this->withoutCar = true;
}

void LaneDetector::manageLaneDetector(std::shared_ptr<cv::Mat> frame)
{
    if (frame->empty()) {
        LogManager::logErrorMessage(ErrorType::IMAGE_ERROR, "No image");
        throw runtime_error("No image!");
    }

    this->image = frame;
    this->imgCols = image->cols;
    this->imgRows = image->rows;

    // Determine if video is taken during daytime or not
    bool isDay = isDayTime();

    // Filter image
    Mat filteredIMG = filterColors(isDay);

    // Apply grayscale
    Mat gray = applyGrayscale(filteredIMG);

    // Apply Gaussian blur
    Mat gBlur = applyGaussianBlur(gray);

    // Find edges
    Mat edges = applyCanny(gBlur);

    // Create mask (Region of Interest)
    Mat maskedIMG = RegionOfInterest(edges);

    // Detect straight lines and draw the lanes if possible
    std::vector<Vec4i> linesP = houghLines(maskedIMG, *image, false);

    // lanes = drawLanes(linesP, check);
    bool isdraw = drawLanes(linesP);

    // No path was found and there is no previous path, default drawing
    if (!isdraw && first) {
        this->drawingInfo.rightX1 = 1207;
        this->drawingInfo.rightX2 = 754;
        this->drawingInfo.leftX1 = 81;
        this->drawingInfo.leftX2 = 505;
        this->drawingInfo.y1 = imgRows;
        this->drawingInfo.y2 = imgRows * (1 - 0.4);
        if (this->withoutCar) {
            cutCar();
        }
    }

    this->first = false;
}
bool LaneDetector::isDayTime()
{
    /* In general, daytime images/videos require different
  color filters than nighttime images/videos. For example, in darker light it is
  better to add a gray color filter in addition to the white and yellow one */

    Scalar s = mean(*image);  // Mean pixel values

    /* Cut off values by looking at the mean pixel values of
  multiple daytime and nighttime images */
    if (s[0] < 30 || s[1] < 33 && s[2] < 30) {
        return false;
    }

    return true;
}

Mat LaneDetector::filterColors(bool isDayTime)
{
    Mat hsv, whiteMask, whiteImage, yellowMask, yellowImage, whiteYellow;

    // White mask
    std::vector<int> lowerWhite = {130, 130, 130};
    std::vector<int> upperWhite = {255, 255, 255};
    inRange(*image, lowerWhite, upperWhite, whiteMask);
    bitwise_and(*image, *image, whiteImage, whiteMask);

    // Yellow mask
    cvtColor(*image, hsv, COLOR_BGR2HSV);
    std::vector<int> lowerYellow = {20, 100, 110};
    std::vector<int> upperYellow = {30, 180, 240};
    inRange(hsv, lowerYellow, upperYellow, yellowMask);
    bitwise_and(*image, *image, yellowImage, yellowMask);

    // Blend yellow and white together
    addWeighted(whiteImage, 1., yellowImage, 1., 0., whiteYellow);

    // Add gray filter if image is not taken during the day
    if (!isDayTime) {
        // Gray mask
        Mat grayMask, grayImage, grayAndWhite, dst;
        std::vector<int> lowerGray = {80, 80, 80};
        std::vector<int> upperGray = {130, 130, 130};
        inRange(*image, lowerGray, upperGray, grayMask);
        bitwise_and(*image, *image, grayImage, grayMask);

        // Blend gray, yellow and white together and return the result
        addWeighted(grayImage, 1., whiteYellow, 1., 0., dst);
        return dst;
    }

    // Return white and yellow mask if image is taken during the day
    return whiteYellow;
}

Mat LaneDetector::applyGrayscale(Mat source)
{
    Mat dst;
    cvtColor(source, dst, COLOR_BGR2GRAY);

    return dst;
}

Mat LaneDetector::applyGaussianBlur(Mat source)
{
    Mat dst;
    GaussianBlur(source, dst, Size(3, 3), 0);

    return dst;
}

Mat LaneDetector::applyCanny(Mat source)
{
    Mat dst;

    if (CAR_IN_IMAGE >= source.rows) {
        LogManager::logErrorMessage(ErrorType::SIZE_ERROR,
                                    "car's size is bigger then image");
        throw runtime_error("car's size is bigger then image");
    }

    Mat roi = source(Rect(0, 0, source.cols, source.rows - CAR_IN_IMAGE));

    Mat extendedImage(CAR_IN_IMAGE, roi.cols, roi.type(), Scalar(0, 0, 0));

    vconcat(roi, extendedImage, roi);

    Canny(roi, dst, 50, 150);

    return dst;
}

Mat LaneDetector::RegionOfInterest(Mat source)
{
    /* In an ideal situation, the ROI should only contain the road lanes.
  We want to filter out all the other stuff, including things like arrow road
  markings. We try to achieve that by creating two trapezoid masks: one big
  trapezoid and a smaller one. The smaller one goes inside the bigger one. The
  pixels in the space between them will be kept and all the other pixels will be
  masked. If it goes well, the space between the two trapezoids contains only
  the lanes. */

    // Parameters big trapezoid

    // Width of bottom edge of trapezoid, expressed as percentage of image width
    float trapezoidBottomWidth = 1.0;
    // Above comment also applies here, but then for the top edge of trapezoid
    float trapezoidTopWidth = 0.07;
    // Height of the trapezoid expressed as percentage of image height
    float trapezoidHeight = 0.5;

    // Parameters small trapezoid

    // will be added to trapezoidBottomWidth to create a less wide bottom edge
    float smallBottomWidth = 0.45;
    // Multiply the percentage trapoezoidTopWidth to create a less wide top edge
    float smallTopWidth = 0.3;
    // Height of the small trapezoid expressed as percentage of height of big trapezoid
    float smallHeight = 1.0;
    // Make the trapezoids float just above the bottom edge of the image
    float bar = 0.97;

    // Vector which holds all the points of the two trapezoids
    std::vector<Point> pts;

    // Large trapezoid
    pts.push_back(cv::Point((source.cols * (1 - trapezoidBottomWidth)) / 2,
                            source.rows * bar));  // Bottom left
    pts.push_back(
        cv::Point((source.cols * (1 - trapezoidTopWidth)) / 2,
                  source.rows - source.rows * trapezoidHeight));  // Top left
    pts.push_back(
        cv::Point(source.cols - (source.cols * (1 - trapezoidTopWidth)) / 2,
                  source.rows - source.rows * trapezoidHeight));  // Top right
    pts.push_back(
        cv::Point(source.cols - (source.cols * (1 - trapezoidBottomWidth)) / 2,
                  source.rows * bar));  // Bottom right

    // Small trapezoid
    pts.push_back(cv::Point(
        (source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2,
        source.rows * bar));  // Bottom left
    pts.push_back(
        cv::Point((source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2,
                  source.rows - source.rows * trapezoidHeight *
                                    smallHeight));  // Top left
    pts.push_back(cv::Point(
        source.cols -
            (source.cols * (1 - trapezoidTopWidth * smallTopWidth)) / 2,
        source.rows -
            source.rows * trapezoidHeight * smallHeight));  // Top right
    pts.push_back(cv::Point(
        source.cols -
            (source.cols * (1 - trapezoidBottomWidth + smallBottomWidth)) / 2,
        source.rows * bar));  // Bottom right

    // Create the mask
    Mat mask = Mat::zeros(source.size(), source.type());
    fillPoly(mask, pts, Scalar(255, 255, 255));

    /* Put the mask over the source image,
  return an all black image, except for the part where the mask image
  has nonzero pixels: all the pixels in the space between the two trapezoids */
    Mat maskedImage;
    bitwise_and(source, mask, maskedImage);

    return maskedImage;
}

std::vector<Vec4i> LaneDetector::houghLines(Mat canny, Mat source,
                                            bool drawHough)
{
    // Distance resolution in pixels of the Hough grid
    double rho = 2;
    // Angular resolution in radians of the Hough grid
    double theta = 1 * M_PI / 180;
    // Minimum number of votes (intersections in Hough grid cell)
    int thresh = 15;
    // Minimum number of pixels making up a line
    double minLineLength = 10;
    // Maximum gap in pixels between connectable line segments
    double maxGapLength = 20;

    std::vector<Vec4i> linesP;  // Will hold the results of the detection
    HoughLinesP(canny, linesP, rho, theta, thresh, minLineLength, maxGapLength);
    if (drawHough) {
        for (size_t i = 0; i < linesP.size(); i++) {
            Vec4i l = linesP[i];
            line(source, Point(l[0], l[1]), Point(l[2], l[3]),
                 Scalar(0, 0, 255), 3, LINE_AA);
        }
        imshow("Hough Lines", source);
        waitKey(5);
    }

    return linesP;
}

bool LaneDetector::drawLanes(std::vector<Vec4i> lines)
{
    if (lines.size() == 0) {
        // There are no lines, use the previous lines
        return false;
    }

    // variables for current paths- Set drawing lanes to true
    bool drawRightLane = true;
    bool drawLeftLane = true;

    // Find lines with a slope higher than the slope threshold
    float slopeThreshold = 0.5;
    std::vector<float> slopes;
    std::vector<Vec4i> goodLines;

    for (int i = 0; i < lines.size(); i++) {
        /* Each line is represented by a 4-element vector (x_1, y_1, x_2, y_2),
    where (x_1,y_1) is the line's starting point and (x_2, y_2) is the ending
    point */
        Vec4i l = lines[i];

        double slope;
        // Calculate the slope
        if (l[2] - l[0] == 0) {  // Avoid division by zero
            slope = 999;         // Infinite slope
        }
        else {
            slope = (l[3] - l[1]) / (l[2] / l[0]);
        }

        if (abs(slope) > slopeThreshold) {
            slopes.push_back(slope);
            goodLines.push_back(l);
        }
    }

    /* Split the good lines into two categories: right and left
  The right lines have a positive slope and the left lines have a negative slope */

    // Outlines' lanes.
    std::vector<Vec4i> rightLines;
    std::vector<Vec4i> leftLines;
    int imgCenter = imgCols / 2;

    for (int i = 0; i < slopes.size(); i++) {
        if (slopes[i] > 0 && goodLines[i][0] > imgCenter &&
            goodLines[i][2] > imgCenter) {
            rightLines.push_back(goodLines[i]);
        }
        if (slopes[i] < 0 && goodLines[i][0] < imgCenter &&
            goodLines[i][2] < imgCenter) {
            leftLines.push_back(goodLines[i]);
        }
    }

    /* form two lane lines out of all the lines we've detected.
  A line is defined as 2 points: a starting point and an ending point.
  Our goal at this step is to use linear regression to find the two
  best fitting lines */

    // Define the vervs here. need them in the condition.
    int y1;
    int y2;
    int rightX1;
    int rightX2;
    int leftX1;
    int leftX2;

    // right and left lanes
    std::vector<int> rightLinesX, rightLinesY, leftLinesX, leftLinesY;
    double rightB1, rightB0, leftB1, leftB0;  // Slope and intercept

    if (rightLines.size() != 0 && leftLines.size() != 0) {
        // We start with the right side points
        for (int i = 0; i < rightLines.size(); i++) {
            // X of starting point of line
            rightLinesX.push_back(rightLines[i][0]);
            // X of ending point of line
            rightLinesX.push_back(rightLines[i][2]);
            // Y of starting point of line
            rightLinesY.push_back(rightLines[i][1]);
            // Y of ending point of line
            rightLinesY.push_back(rightLines[i][3]);
        }

        if (rightLinesX.size() > 0) {
            std::vector<double> coefRight = estimateCoefficients<int, double>(
                rightLinesX, rightLinesY);  // y = b1x + b0
            rightB1 = coefRight[0];
            rightB0 = coefRight[1];
        }
        else {
            LogManager::logDebugMessage(DebugType::DRAW_PREV,
                                        "drawRightLane false");
            return false;
        }

        for (int i = 0; i < leftLines.size(); i++) {
            // X of starting point of line
            leftLinesX.push_back(leftLines[i][0]);
            // X of ending point of line
            leftLinesX.push_back(leftLines[i][2]);
            // Y of starting point of line
            leftLinesY.push_back(leftLines[i][1]);
            // Y of ending point of line
            leftLinesY.push_back(leftLines[i][3]);
        }
        if (leftLinesX.size() > 0) {
            std::vector<double> coefLeft =
                estimateCoefficients<int, double>(leftLinesX, leftLinesY);
            leftB1 = coefLeft[0];
            leftB0 = coefLeft[1];
        }
        else {
            LogManager::logDebugMessage(DebugType::DRAW_PREV,
                                        "drawLeftLane false");
            return false;
        }

        /* Now we need to find the two points for the right and left lane:
    starting points and ending points */

        // Y coordinate of starting point of both the left and right lane
        y1 = imgRows;

        /* 0.5 = trapezoidHeight (see RegionOfInterest), we set the y coordinate of
    the ending point below the trapezoid height (0.4) to draw shorter lanes. I
    think that looks nicer. */

        // Y coordinate of ending point of both the left and right lane
        y2 = imgRows * (1 - 0.4);

        // X coordinate of starting point of right lane
        rightX1 = (y1 - rightB0) / rightB1;
        // X coordinate of ending point of right lane
        rightX2 = (y2 - rightB0) / rightB1;
        // X coordinate of starting point of left lane
        leftX1 = (y1 - leftB0) / leftB1;
        // X coordinate of ending point of left lane
        leftX2 = (y2 - leftB0) / leftB1;

        /* If the end point of the paths exceed the boundaries of
     the image excessively,  return prev drawings. */
        if (!first && (rightX2 < leftX2 || leftX2 > rightX2 || rightX1 < 0 ||
                       rightX2 > imgCols || leftX1 > imgCols || leftX2 < 0 ||
                       rightX1 > imgRows + 350 || leftX1 < -350 ||
                       rightX1 - leftX1 < 300)) {
            LogManager::logDebugMessage(
                DebugType::DRAW_PREV,
                "points outside the boundaries of the image");
            return false;
        }

        else {
            LogManager::logDebugMessage(DebugType::PRINT, "current path");
        }

        double angleThreshold = 45.0;  // 45 degree angle
        if (drawRightLane && drawLeftLane) {
            double angle =
                atan(abs((leftB1 - rightB1) / (1 + leftB1 * rightB1))) *
                (180.0 / CV_PI);

            if (angle > angleThreshold && !first) {
                LogManager::logDebugMessage(DebugType::DRAW_PREV,
                                            "angle > angleThreshold");
                return false;
            }
        }
    }

    else {
        LogManager::logDebugMessage(DebugType::DRAW_PREV, "missLine");
        return false;
    }

    // Draw paths
    Mat mask = Mat::zeros(image->size(), image->type());

    drawingInfo.rightX1 = rightX1;
    drawingInfo.rightX2 = rightX2;
    drawingInfo.leftX1 = leftX1;
    drawingInfo.leftX2 = leftX2;
    drawingInfo.y1 = y1;
    drawingInfo.y2 = y2;

    if (this->withoutCar) {
        cutCar();
    }
    cout << "cuurent" << endl;
    LogManager::logDebugMessage(DebugType::PRINT, "Using current path");

    return true;
}

void LaneDetector::drawLane(shared_ptr<Mat> img)
{
    // Draw a transverse line 120 pixels above the bottom of the image
    cv::line(*img, cv::Point(0, img->rows - CAR_IN_IMAGE),
             cv::Point(img->cols, img->rows - CAR_IN_IMAGE),
             cv::Scalar(255, 255, 255), 4);

    // Draw the lanes on the image using the updated values
    cv::line(*img, cv::Point(drawingInfo.rightX1, drawingInfo.y1),
             cv::Point(drawingInfo.rightX2, drawingInfo.y2),
             cv::Scalar(0, 255, 0), 7);
    cv::line(*img, cv::Point(drawingInfo.leftX1, drawingInfo.y1),
             cv::Point(drawingInfo.leftX2, drawingInfo.y2),
             cv::Scalar(255, 0, 0), 7);

    // Fill the area between the lanes
    Point prevPts[4] = {Point(drawingInfo.leftX1, drawingInfo.y1),
                        Point(drawingInfo.leftX2, drawingInfo.y2),
                        Point(drawingInfo.rightX2, drawingInfo.y2),
                        Point(drawingInfo.rightX1, drawingInfo.y1)};
    Mat mask = Mat::zeros(img->size(), img->type());
    fillConvexPoly(mask, prevPts, 4, Scalar(235, 229, 52));

    // Blend the mask and image together
    addWeighted(*img, 0.9, mask, 0.3, 0.0, *img);
}

void LaneDetector::cutCar()
{
    float rightSlope = static_cast<float>(drawingInfo.y2 - drawingInfo.y1) /
                       (drawingInfo.rightX2 - drawingInfo.rightX1);
    float leftSlope = static_cast<float>(drawingInfo.y2 - drawingInfo.y1) /
                      (drawingInfo.leftX2 - drawingInfo.leftX1);

    // Acoording to straight formula: y = slope*x + d.
    float dRight = drawingInfo.y1 - rightSlope * drawingInfo.rightX1;
    float dLeft = drawingInfo.y1 - leftSlope * drawingInfo.leftX1;
    drawingInfo.y1 -= CAR_IN_IMAGE;

    // Finding X by placing Y in the equation of the straight line
    drawingInfo.rightX1 =
        static_cast<int>((drawingInfo.y1 - dRight) / rightSlope);
    drawingInfo.leftX1 = static_cast<int>((drawingInfo.y1 - dLeft) / leftSlope);
}

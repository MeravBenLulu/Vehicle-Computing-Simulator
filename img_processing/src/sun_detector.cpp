#include "sun_detector.h"

using namespace std;
using namespace cv;

void SunDetector::detectSun(const std::shared_ptr<cv::Mat> &frame)
{
    isSun = false;
    // Convert the frame to grayscale for easier processing
    Mat image;
    cvtColor(*frame, image, COLOR_BGR2GRAY);
    // Calculate the histogram of the grayscale image
    Mat histogram;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    calcHist(&image, 1, 0, Mat(), histogram, 1, &histSize, &histRange);
    // Compute the cumulative distribution function (CDF) from the histogram
    Mat cdf;
    histogram.copyTo(cdf);
    for (int i = 1; i < histSize; i++) {
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    }
    cdf /= cdf.at<float>(histSize - 1);  // Normalize the CDF
    // Determine a threshold based on the 95th percentile of pixel intensities
    double minVal, maxVal;
    minMaxLoc(image, &minVal, &maxVal);  // Find min and max pixel intensities
    Mat percentileThreshold;
    threshold(image, percentileThreshold, maxVal * 0.95, 255, THRESH_BINARY);
    int thresholdArea = cv::countNonZero(percentileThreshold);
    // Detect contours in the thresholded image (for bright regions)
    std::vector<std::vector<Point>> contours;
    findContours(percentileThreshold, contours, RETR_EXTERNAL,
                 CHAIN_APPROX_SIMPLE);
    // Find the largest contour in terms of area (which could be the sun)
    double maxArea = 0;
    int maxAreaIdx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }
    // If a valid contour was found, proceed with further analysis
    if (maxAreaIdx != -1) {
        // Draw a circle around the largest contour to visualize it
        minEnclosingCircle(contours[maxAreaIdx], center, radius);
        // Calculate image gradients (Sobel) to analyze the smoothness of the region
        Mat gradX, gradY;
        Sobel(image, gradX, CV_32F, 1, 0);  // Gradient in X direction
        Sobel(image, gradY, CV_32F, 0, 1);  // Gradient in Y direction
        // Calculate the gradient magnitude (intensity of edges)
        Mat gradientMag;
        magnitude(gradX, gradY, gradientMag);  // sqrt(gradX^2 + gradY^2)
        // Compute the average gradient magnitude in the bright region (contour
        // area)
        Scalar meanGradient = mean(gradientMag, percentileThreshold);
        // If the gradient magnitude is low, the area is likely smooth, like the sun
        if (meanGradient[0] < 7 &&
            thresholdArea < 35000)  // Adjust this threshold as needed
        {
            isSun = true;
        }
    }
}

void SunDetector::drowSun(std::shared_ptr<cv::Mat> &frame)
{
    if (isSun) {
        // Draw a green circle around the detected sun region
        circle(*frame, center, static_cast<int>(radius), Scalar(0, 255, 0), 2);
    }
}

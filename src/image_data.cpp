#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include "../include/image_data.hpp"

using namespace std;

ImageData::ImageData(cv::String imagePath, cv::Matx33f intrinsicMat, glm::vec3 lineCol, const cv::Mat* transformation) {
    cout << "Imagedata constructor" << endl;
    image = cv::imread(imagePath, cv::IMREAD_COLOR);
    cameraIntrinsic = intrinsicMat;
    cout << "cameraIntrinsic = intrinsicMat;" << endl;

    //Detect features in the image.
    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
    cout << "Set up Detector" << endl;

    detector->detectAndCompute(image, cv::Mat(), image_keypoints, image_descriptors); 
    cout << "Detected" << endl;

    // cout << "tranformation parameter: " << transformation << endl;

    worldTransformation = *transformation;

    cout << "worldTransformation = transformation;" << endl;

    lineColour = lineCol;
}
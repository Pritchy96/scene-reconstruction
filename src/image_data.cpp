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

ImageData::ImageData(cv::String imagePath, cv::Mat intrinsicMat, glm::vec3 lineCol, glm::mat4 transform) {
    image = cv::imread(imagePath, cv::IMREAD_COLOR);
    cameraIntrinsic = intrinsicMat;

    //Detect features in the image.
    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create();
    detector->detectAndCompute(image, cv::Mat(), image_keypoints, image_descriptors); 

    worldTransformation = transform;
    lineColour = lineCol;
}
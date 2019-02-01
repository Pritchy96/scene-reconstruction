
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include "../include/image_data.hpp"

using namespace std;

ImageData::ImageData(cv::String imagePath, cv::Matx33d intrinsicMat, cv::InputArray translation, cv::InputArray rotation) {
    worldTranslation = translation.getMat();
    worldRotation = rotation.getMat();

    SetupAndDetectKeyPoints(imagePath, intrinsicMat);
}

ImageData::ImageData(cv::String imagePath, cv::Matx33d intrinsicMat, cv::Mat _transformation) {
    CV_Assert(_transformation.type() == CV_64F);
    worldRotation = _transformation.rowRange(0,3).colRange(0,3);
    worldTranslation = _transformation.rowRange(0,3).col(3);

    SetupAndDetectKeyPoints(imagePath, intrinsicMat);
}

void ImageData::SetupAndDetectKeyPoints(cv::String imagePath, cv::Matx33d intrinsicMat) {
    image = cv::imread(imagePath, cv::IMREAD_ANYCOLOR);
    cameraIntrinsic = intrinsicMat;

    //Detect features in the image.
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(5000);
    orb->detectAndCompute(image, cv::Mat(), image_keypoints, image_descriptors);

    if (DEBUG_LOG) {
        cv::Mat output_image_keypoints;
        drawKeypoints( image, image_keypoints, output_image_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
        imshow("Detected Keypoints", output_image_keypoints );
        cv::waitKey(0);
    }
}
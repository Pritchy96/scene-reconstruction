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

ImageData::ImageData(cv::String imagePath, cv::Matx33f intrinsicMat, glm::vec3 lineCol, cv::Matx34f transformation) {
    image = cv::imread(imagePath, cv::IMREAD_ANYCOLOR);
    cameraIntrinsic = intrinsicMat;

    //Detect features in the image.
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(100000);
    // cout << "Set up Detector" << endl;
    orb->detectAndCompute(image, cv::Mat(), image_keypoints, image_descriptors);
    // cout << "Detected" << endl;

    //DEBUG
    cv::Mat output_image_keypoints;
    drawKeypoints( image, image_keypoints, output_image_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
    imshow("Detected Keypoints", output_image_keypoints );


    // cv::Ptr<cv::xfeatures2d> detector = cv::xfeatures2d::SiftFeatureDetector::create();
    // detector->detectAndCompute(image, cv::Mat(), image_keypoints, image_descriptors); 

    worldTransformation = transformation;
    lineColour = lineCol;
}
#ifndef IMAGEDATA_HPP
#define IMAGEDATA_HPP

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


    class ImageData {
        public:
            cv::Mat image,
                image_descriptors;  //Characteristics of the keypoints
            cv::Matx33d cameraIntrinsic;
            std::vector<cv::KeyPoint> image_keypoints;
            cv::Mat worldTranslation;
            cv::Mat worldRotation;
            glm::vec3 lineColour;

            ImageData(cv::String imagePath, cv::Matx33d intrinsicMat, cv::InputArray translation, cv::InputArray rotation);
            ImageData(cv::String imagePath, cv::Matx33d intrinsicMat, cv::Mat _transformation);
    };

#endif
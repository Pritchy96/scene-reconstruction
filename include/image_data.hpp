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
            cv::Mat cameraIntrinsic;
            std::vector<cv::KeyPoint> image_keypoints;
            cv::Mat worldTransform;
            glm::vec3 lineColour;
            static const bool DEBUG_LOG = false;

            // ImageData(cv::Mat _image, cv::Mat intrinsicMat, cv::InputArray translation, cv::InputArray rotation);
            ImageData(cv::Mat _image, cv::Mat _cameraIntrinsic, cv::Mat _worldTransform);
        private:
            void SetupAndDetectKeyPoints();
    };

#endif
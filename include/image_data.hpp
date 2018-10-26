#ifndef IMAGEDATA_HPP
#define IMAGEDATA_HPP

#include <vector>

    class ImageData {
        public:
            cv::Mat image,
                image_descriptors;  //Characteristics of the keypoints
            cv::Matx33f cameraIntrinsic;
            std::vector<cv::KeyPoint> image_keypoints;
            cv::Mat worldTransformation;
            glm::vec3 lineColour;

            ImageData(cv::String imagePath, cv::Matx33f intrinsicMat, glm::vec3 lineCol, const cv::Mat* transformation);
    };

#endif
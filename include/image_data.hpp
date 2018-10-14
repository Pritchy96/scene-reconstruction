#ifndef IMAGEDATA_HPP
#define IMAGEDATA_HPP

#include <vector>

    class ImageData {
        public:
            cv::Mat image, 
                cameraIntrinsic,
                image_descriptors;  //Characteristics of the keypoints
            std::vector<cv::KeyPoint> image_keypoints;
            glm::mat4 worldTransformation;
            glm::vec3 lineColour;

            ImageData(cv::String imagePath, cv::Mat cameraIntrinsic, glm::vec3 lineCol, glm::mat4 transform = glm::mat4(1.0f));
    };
#endif
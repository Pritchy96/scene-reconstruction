#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glm/gtc/matrix_transform.hpp"
#define GLM_ENABLE_EXPERIMENTAL //gtx = gt eXperimental?
#include <glm/gtx/string_cast.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "../include/image_data.hpp"
#include "../include/image_data_set.hpp"

using namespace std;

void fromCV2GLM(const cv::Mat& cvmat, glm::mat3* glmmat) {
    //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
    if (cvmat.cols != 3|| cvmat.rows != 3 || cvmat.type() != CV_32FC1) {
        cout << "Matrix conversion error! (3x3)" << endl;
        return;
    }
    memcpy(glm::value_ptr(*glmmat), cvmat.data, 12 * sizeof(float));
}

void fromGLM2CV(const glm::mat3& glmmat, cv::Mat* cvmat) {
    //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
    if (cvmat->cols != 3 || cvmat->rows != 3) {
        (*cvmat) = cv::Mat(3, 3, CV_32F);
    }
    memcpy(cvmat->data, glm::value_ptr(glmmat), 12 * sizeof(float));
}

void fromCV2GLM(const cv::Mat& cvmat, glm::mat4* glmmat) {
    //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
    if (cvmat.cols != 4|| cvmat.rows != 4 || cvmat.type() != CV_32FC1) {
        cout << "Matrix conversion error! (4x4) " << cvmat.cols << endl;
        return;
    }
    memcpy(glm::value_ptr(*glmmat), cvmat.data, 16 * sizeof(float));
}

void fromGLM2CV(const glm::mat4& glmmat, cv::Mat* cvmat) {
    //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
    if (cvmat->cols != 4 || cvmat->rows != 4) {
        (*cvmat) = cv::Mat(4, 4, CV_32F);
    }
    memcpy(cvmat->data, glm::value_ptr(glmmat), 16 * sizeof(float));
}

ImageDataSet::ImageDataSet(ImageData *img1, ImageData *img2) {
    //https://docs.opencv.org/3.1.0/d5/d6f/tutorial_feature_flann_matcher.html
    
    image1 = img1; image2 = img2;

    FindMatchingFeatures(true);
    relativeTransformation = EstimateRelativePose();

    if (!valid) {return;} //No Essential Matrix found.

    //Calculate image2 world transform.
    if (image1->worldTransformation == glm::mat4(1.0f)) {
        //If we're on the first image (no world tranform for previous image), then worldTransform = relativeTransform.
        image2->worldTransformation = relativeTransformation;
    } else {
        //Otherwise, the world tranform is the sum translation of all previous relative transforms to get the world transform of image1
        //..plus the relative transform to get from image1 to image2
        image2->worldTransformation = -image1->worldTransformation * relativeTransformation;
    }
} 

void ImageDataSet::FindMatchingFeatures(bool displayResults) {
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> image_matches;
    matcher.match(image1->image_descriptors, image2->image_descriptors, image_matches);

    double max_dist = 0; double min_dist = 100;
    //Quick calculation of max and min distances between keypoints
    //Taken from https://docs.opencv.org/3.1.0/d5/d6f/tutorial_feature_flann_matcher.html
    //TODO: find good thresholds for this.
    for( int i = 0; i < image1->image_descriptors.rows; i++ ) {
        double dist = image_matches[i].distance;

        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    for (int i = 0; i < (int)image_matches.size(); i++) {
        if (image_matches[i].distance <= cv::max(2*min_dist, 0.02)) {
            goodMatches.push_back(image_matches[i]);
            points2.push_back(image2->image_keypoints[image_matches[i].trainIdx].pt);
            points1.push_back(image1->image_keypoints[image_matches[i].queryIdx].pt);
        }
    }

    if (displayResults) DisplayMatches(); 
}

glm::mat4 ImageDataSet::EstimateRelativePose() {
    cv::Mat essentialMat = cv::findEssentialMat(cv::Mat(points1), cv::Mat(points2), image1->cameraIntrinsic, cv::RANSAC, 0.1f, 1.0);
    cv::Mat fundamentalMat = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), cv::FM_RANSAC);

    //OpenCV returns a non 3x3 matrix if it can't derive an Essential Matrix.
    if (essentialMat.cols != 3 || essentialMat.rows != 3) {
        cout << "Not enough matched points to derive EssentialMatrix" << endl;
        valid = false;
        return glm::mat4(0.0f); //TODO: check this code path.
    }

    cv::Mat cvRotation, cvTranslation;

    // cv::decomposeEssentialMat(essentialMat, cvRotation1, cvRotation2, cvTranslation);
    cv::recoverPose(essentialMat, points1, points2, image1->cameraIntrinsic, cvRotation, cvTranslation);

    //Construct a transformation mat from a translation and a rotation mat.
    cv::Mat cv_rt = cv::Mat::eye(4, 4, CV_32FC1);
    cvRotation.copyTo(cv_rt.rowRange(0,3).colRange(0,3));
    cvTranslation.copyTo(cv_rt.rowRange(0,3).col(3));

    glm::mat4 glm_rt;
    fromCV2GLM(cv_rt, &glm_rt);

    return glm_rt;
}

void ImageDataSet::DisplayMatches() {
    cv::drawMatches(image1->image, image1->image_keypoints, image2->image, image2->image_keypoints,
                goodMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //Show detected matches in an image viewer for debug purposes. 
    imshow("Good Matches", img_matches);
    cv::waitKey(0); //Wait for a key to be hit to exit viewer.
}

void ImageDataSet::DrawEpipolarLines(cv::Mat& image_out, const cv::Mat& image1, 
    const cv::Mat& image2, cv::Mat fundamentalMat,
    vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, 
    int whichImage) // image to compute epipolar lines in
    {
        vector<cv::Vec3f> lines1;

        // Compute corresponding epipolar lines
        cv::computeCorrespondEpilines(cv::Mat(points1), // image points
        whichImage, // in image 1 (can also be 2)
        fundamentalMat,
        lines1); // vector of epipolar lines

        // for all epipolar lines
        for (vector<cv::Vec3f>::const_iterator it = lines1.begin(); it!=lines1.end(); ++it) {
            // Draw the line between first and last column
            cv::line(image_out,
                cv::Point(0,-(*it)[2]/(*it)[1]),
                cv::Point(image2.cols,-((*it)[2] + (*it)[0]*image2.cols)/(*it)[1]),
                cv::Scalar(255,255,255)
            );
        }
    }


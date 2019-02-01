#include <iostream>
#include <vector>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glm/gtc/matrix_transform.hpp"
#define GLM_ENABLE_EXPERIMENTAL //gtx = gt eXperimental?
#include <glm/gtx/string_cast.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "../include/image_data.hpp"
#include "../include/image_data_set.hpp"

using namespace std;

// void fromCV2GLM(const cv::Mat& cvmat, glm::mat3* glmmat) {
//     //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
//     if (cvmat.cols != 3|| cvmat.rows != 3 || cvmat.type() != CV_32FC1) {
//         cout << "Matrix conversion error! (3x3)" << endl;
//         return;
//     }
//     memcpy(glm::value_ptr(*glmmat), cvmat.data, 12 * sizeof(float));
// }

// void fromGLM2CV(const glm::mat3& glmmat, cv::Mat* cvmat) {
//     //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
//     if (cvmat->cols != 3 || cvmat->rows != 3) {
//         (*cvmat) = cv::Mat(3, 3, CV_32F);
//     }
//     memcpy(cvmat->data, glm::value_ptr(glmmat), 12 * sizeof(float));
// }

// void fromCV2GLM(const cv::Mat& cvmat, glm::mat4* glmmat) {
//     //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
//     if (cvmat.cols != 4|| cvmat.rows != 4 || cvmat.type() != CV_32FC1) {
//         cout << "Matrix conversion error! (4x4) " << cvmat.cols << endl;
//         return;
//     }
//     memcpy(glm::value_ptr(*glmmat), cvmat.data, 16 * sizeof(float));
// }

// void fromGLM2CV(const glm::mat4& glmmat, cv::Mat* cvmat) {
//     //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
//     if (cvmat->cols != 4 || cvmat->rows != 4) {
//         (*cvmat) = cv::Mat(4, 4, CV_32F);
//     }
//     memcpy(cvmat->data, glm::value_ptr(glmmat), 16 * sizeof(float));
// }

ImageDataSet::ImageDataSet(ImageData *img1, ImageData *img2) {
    image1 = img1; image2 = img2;

    FindMatchingFeatures(true);
    EstimateRelativePose();

    // if (!valid) {return;} //No Essential Matrix found.
    //Calculate image2 world transform.
    if (!valid) {
         //If we can't decompose an essential matrix, just set the transform to the same one as the last image.
         //...See if the Bundle Adjustment will compensate.
         cout << "unable to decompose Essential matrix" << endl;
        image2->worldTranslation = image1->worldTranslation;
        image2->worldRotation = image1->worldRotation;

    }
    else if (cv::countNonZero(image1->worldRotation) == 0 && cv::countNonZero(image1->worldTranslation) == 0) {
        //If we're on the first image (no world tranform for previous image), then worldTransform = relativeTransform.
        image2->worldTranslation = image1->worldTranslation;
        image2->worldRotation = image1->worldRotation;
    } else {
        //Otherwise, the world tranform is the sum translation of all previous relative transforms to get the world transform of image1
        //..plus the relative transform to get from image1 to image2

        //Construct a transformation mat from a translation and a rotation mat.
        cv::Mat i1WorldTransformation = cv::Mat::eye(3, 4, CV_64F),
                    relativeTransformation = cv::Mat::eye(3, 4, CV_64F);       
        image1->worldRotation.copyTo(i1WorldTransformation.rowRange(0,3).colRange(0,3));
        image1->worldTranslation.copyTo(i1WorldTransformation.rowRange(0,3).colRange(3, 4));
        relativeRotation.copyTo(relativeTransformation.rowRange(0,3).colRange(0,3));
        relativeTranslation.copyTo(relativeTransformation.rowRange(0,3).colRange(3,4));

        //Multiply the two transforms
        cv::Mat result = cv::Mat::eye(3, 4, CV_64F);        
        cv::multiply(i1WorldTransformation, relativeTransformation, result);

        //Split back into separate rotations/translations.
        image2->worldRotation = result.rowRange(0,3).colRange(0,3);
        image2->worldTranslation = result.rowRange(0,3).col(3);
        cout << endl << endl;
    }
} 

void ImageDataSet::FindMatchingFeatures(bool displayResults) {
    vector<cv::Point2f> initialMatchesP1, initialMatchesP2;

    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher.knnMatch(image1->image_descriptors, image2->image_descriptors, knn_matches, 2);
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    // std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i].size() > 0 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            goodMatches.push_back(knn_matches[i][0]);
            initialMatchesP2.push_back(image2->image_keypoints[knn_matches[i][0].trainIdx].pt);
            initialMatchesP1.push_back(image1->image_keypoints[knn_matches[i][0].queryIdx].pt);
        }
    }

    

    // Filter bad matches using fundamental matrix constraint
    cv::findFundamentalMat(initialMatchesP2, initialMatchesP1, cv::FM_RANSAC, 3.0, 0.99, mask);

    cout << mask << endl;
    for(int i = 0; i < mask.rows; i++) {
        if(mask.at<unsigned char>(i)){
            points1.push_back(initialMatchesP1[i]);
            points2.push_back(initialMatchesP2[i]);

            // cv::line(canvas, src[k], dst[k] + Point2f(0, img_pose_i.img.rows), Scalar(0, 0, 255), 2);
        }
    }

    // std::vector<cv::DMatch> image_matches;

//     cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
//     matcher->match(image1->image_descriptors, image2->image_descriptors, image_matches);

//     double max_dist = 0; double min_dist = 100;
//     //Quick calculation of max and min distances between keypoints
//     //Taken from https://docs.opencv.org/3.1.0/d5/d6f/tutorial_feature_flann_matcher.html
//     //TODO: find good thresholds for this.
//     for( int i = 0; i < image1->image_descriptors.rows; i++ ) {
//         double dist = image_matches[i].distance;
        
//         if( dist < min_dist ) min_dist = dist;
//         if( dist > max_dist ) max_dist = dist;
//     }

//     for (int i = 0; i < (int)image_matches.size(); i++) {
//         if (image_matches[i].distance <= cv::max(2*min_dist, 0.02)) {
//             goodMatches.push_back(image_matches[i]);
//             points2.push_back(image2->image_keypoints[image_matches[i].trainIdx].pt);
//             points1.push_back(image1->image_keypoints[image_matches[i].queryIdx].pt);
//         }
//     }

    if (displayResults) DisplayMatches(); 
}

void ImageDataSet::EstimateRelativePose() {
    cv::Point2d pp(cv::Mat(image1->cameraIntrinsic).at<double>(0,2), cv::Mat(image1->cameraIntrinsic).at<double>(1,2));
    double focal = cv::Mat(image1->cameraIntrinsic).at<double>(0,0);
        
    // cv::Mat essentialMat = cv::findEssentialMat(cv::Mat(points1), cv::Mat(points2), image1->cameraIntrinsic, cv::RANSAC, 0.99899999, 0.1f, cv::noArray());
    // cv::Mat essentialMat = cv::findEssentialMat(cv::Mat(points1), cv::Mat(points2), image1->cameraIntrinsic, cv::LMEDS, 0.999,  0.1f, mask);
    cv::Mat essentialMat = cv::findEssentialMat(cv::Mat(points1), cv::Mat(points2), focal,
                                                    pp, cv::RANSAC, 0.999, 1.0, mask);

    cout << "mask: " << mask << endl;        //Pass in two lists of image keypoints
        //Feature match
        //Remove poor matches

    // cv::correctMatches(essentialMat, points1, points2, points1, points2);
    //cv::Mat fundamentalMat = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), cv::FM_RANSAC);

    // vector<cv::Point2f> undistortedPoints1, undistortedPoints2;
    // cv::undistortPoints(points1, undistortedPoints1, image1->cameraIntrinsic, cv::noArray());
    // cv::undistortPoints(points2, undistortedPoints2, image2->cameraIntrinsic, cv::noArray());

    //OpenCV returns a non 3x3 matrix if it can't derive an Essential Matrix.
    if (essentialMat.cols != 3 || essentialMat.rows != 3) {
        cout << "Not enough matched points to derive EssentialMatrix" << endl;
        valid = false;
        relativeTranslation = cv::Mat3f::eye(3, 3); //TODO: check this code path.
        relativeRotation = cv::Mat1f::eye(3, 1);
        return;
    }

    vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
    for(int i = 0; i < mask.rows; i++) {
        if(mask.at<unsigned char>(i)){
        inlier_match_points1.push_back(points1[i]);
        inlier_match_points2.push_back(points2[i]);
        }
    }

    if(true) {
        cv::Mat src;
        cv::hconcat(image1->image, image2->image, src);
        for(int i = 0; i < inlier_match_points1.size(); i++) {
        cv::line( src, inlier_match_points1[i],
                    cv::Point2f(inlier_match_points2[i].x + image1->image.cols, inlier_match_points2[i].y),
                    cv::Scalar( 255, 0, 0 ), 1, 0 );
        }
        cv::imwrite("inlier_match_points.png", src);
    }

    mask.release();
    cv::Mat cvRotation, cvTranslation;

    // cv::decomposeEssentialMat(essentialMat, cvRotation1, cvRotation2, cvTranslation);
    // cv::recoverPose(essentialMat, points1, points2, image1, cvRotation, cvTranslation, mask);
    cv::recoverPose(essentialMat, inlier_match_points1, inlier_match_points2, cvRotation, cvTranslation, focal, pp);

    relativeRotation = cvRotation;
    relativeTranslation = cvTranslation;
}

vector<cv::Point3f> ImageDataSet::TriangulatePoints(vector<cv::Point2f> image1Points, vector<cv::Point2f> image2Points) {
    cv::Mat i1WorldTransformation = cv::Mat::eye(3, 4, CV_64FC1);
    image1->worldRotation.copyTo(i1WorldTransformation.rowRange(0,3).colRange(0,3));
    image1->worldTranslation.copyTo(i1WorldTransformation.rowRange(0,3).col(3));

    cv::Mat i2WorldTransformation = cv::Mat::eye(3, 4, CV_64FC1);
    image2->worldRotation.copyTo(i2WorldTransformation.rowRange(0,3).colRange(0,3));
    image2->worldTranslation.copyTo(i2WorldTransformation.rowRange(0,3).col(3));

    cv::Mat cameraIntrinsicDouble;
    cv::Mat(image1->cameraIntrinsic).convertTo(cameraIntrinsicDouble, CV_64F);

    cv::Mat points;
    cv::triangulatePoints(cameraIntrinsicDouble * i1WorldTransformation, cameraIntrinsicDouble * i2WorldTransformation, image1Points, image2Points, points);

    vector<cv::Point3f> points3D;
    for (int i = 0; i < points.cols; i++) {
        vector<cv::Point3f> p3d;
        convertPointsFromHomogeneous(points.col(i).t(), p3d);
        // cout << "x: " << p3d[0].x << ", y: " << p3d[0].y<< ", z: " << p3d[0].z << endl;
        points3D.insert(points3D.end(), p3d.begin(), p3d.end());
    }
    return points3D;
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


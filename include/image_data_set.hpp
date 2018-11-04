#ifndef IMAGEDATASET_HPP
#define IMAGEDATASET_HPP

#include <map>
#include "../include/point2fCompare.hpp"


using namespace std;

    class ImageDataSet {
        public:
            ImageData *image1, *image2;
            vector<cv::Point2f> points1, points2;   //TODO: Rename these to something more appropriate.
            cv::Mat relativeTranslation, relativeRotation;
            vector<cv::DMatch> goodMatches;
            vector<glm::vec3> pointCloud;
            std::map<cv::Point2f, int> visibilityLocations; //TODO: rename
            cv::Mat img_matches;
            bool valid = true;

            // ImageDataSet(ImageData *img1, ImageData *img2);
            ImageDataSet(ImageData *img1, ImageData *img2);
            void DisplayMatches();
            void EstimateRelativePose();
            void FindMatchingFeatures(bool displayResults);
            void DrawEpipolarLines(cv::Mat& image_out, const cv::Mat& image1, 
                const cv::Mat& image2, cv::Mat fundamentalMat,
                vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, 
                int whichImage);
            vector<cv::Point3f> TriangulatePoints(vector<cv::Point2f> image1Points, vector<cv::Point2f> image2Points);
    };
#endif
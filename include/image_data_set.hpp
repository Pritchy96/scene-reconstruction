#ifndef IMAGEDATASET_HPP
#define IMAGEDATASET_HPP

using namespace std;

    class ImageDataSet {
        public:
            ImageData *image1, *image2;
            vector<cv::Point2f> points1, points2;
            glm::mat4 relativeTransformation;
            vector<cv::DMatch> goodMatches;
            cv::Mat img_matches;
            vector<float> image1PointLines, image1PointColours, image2PointLines, image2PointColours, cubePoints;
            bool valid = true;

            ImageDataSet(ImageData *img1, ImageData *img2);
            void DisplayMatches();
            glm::mat4 EstimateRelativePose();
            void FindMatchingFeatures(bool displayResults);
            void DrawEpipolarLines(cv::Mat& image_out, const cv::Mat& image1, 
                const cv::Mat& image2, cv::Mat fundamentalMat,
                vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, 
                int whichImage);
    };
#endif
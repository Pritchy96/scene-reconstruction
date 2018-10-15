#ifndef IMAGEDATASET_HPP
#define IMAGEDATASET_HPP

using namespace std;

    class ImageDataSet {
        public:
            ImageData *image1, *image2;
            vector<cv::Point2f> points1, points2;   //TODO: Rename these to something more appropriate.
            glm::mat4 relativeTransformation;
            vector<cv::DMatch> goodMatches;
            vector<glm::vec3> pointCloud;
            cv::Mat img_matches;
            bool valid = true;

            ImageDataSet(ImageData *img1, ImageData *img2);
            void DisplayMatches();
            glm::mat4 EstimateRelativePose();
            void FindMatchingFeatures(bool displayResults);
            void DrawEpipolarLines(cv::Mat& image_out, const cv::Mat& image1, 
                const cv::Mat& image2, cv::Mat fundamentalMat,
                vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, 
                int whichImage);
            void TriangulatePoints();
    };
#endif
#include <iostream>
#include <numeric>
#include <chrono>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <cvsba/cvsba.h>
#include <GL/glew.h>
#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include "../include/point2fCompare.hpp"
#include "../include/image_data.hpp"
#include "../include/image_data_set.hpp"
#include "../include/render_environment.hpp"
#include "../include/shader.hpp"

using namespace std;
using namespace boost;

const string imageDir = "./bin/data/desk/";
vector<string> acceptedExtensions = {".png", ".jpg", ".PNG", ".JPG"};

const int IMAGE_DOWNSAMPLE = 4; // downsample the image to speed up processing
const double FOCAL_LENGTH = 4308 / IMAGE_DOWNSAMPLE; // focal length in pixels, after downsampling, guess from jpeg EXIF data
const int MIN_LANDMARK_SEEN = 3; // minimum number of camera views a 3d point (landmark) has to be seen to be used

cv::Mat cameraIntrinsic;


//Given with Dataset.
// const cv::Matx33d cameraIntrinsic (3310.400000f, 0.000000f, 320.0000f,
//                                 0.000000f, 3325.500000f, 240.0000f,
//                                 0.000000f, 0.000000f, 1.000000f);

// double initPose[]{-0.08661715685291285200, 0.97203145042392447000, 0.21829465483805316000, -0.97597093004059532000, -0.03881511324024737600,
//     -0.21441803766270939000, -0.19994795321325870000, -0.23162091782017200000, 0.95203517502501223000, -0.0526034704197, 0.023290917003, 0.659119498846};

//Synthetic
// //Given with Dataset.
// const cv::Matx33d cameraIntrinsic (851.01, 0.000, 796.5,
//                                     0.000, 851.01, 352.5,
//                                     0.000, 0.000, 1.000);

// double initPose[]{0.977755, 0, 0.209751,
//                     0.177552, 0.532404, -0.827659,
//                     -0.111672, 0.846489, 0.52056,
//                     0, 0, -610.999
// };

cv::Mat initialPose = cv::Mat::eye(3, 4, CV_64F);

auto oldTime = chrono::steady_clock::now(), newTime = chrono::steady_clock::now();
double deltaT;

vector<ImageData*> images;
map<cv::Point2f, int> previousPairImage2FeaturesToPoints3D; 
//A list of a list of guesses for each 3d Point. Each list gets averaged out to a single 3D point in points3D.
vector<vector<cv::Point3f>> points3DGuesses;   
vector<glm::vec3> points3D;
vector<glm::vec3> cameras3D;
//List of a list of each images' detected features. This is not sparse; imagePoints[0][1] does not have to be equal to imagePoints[1][1] even if they do have a match!
vector<vector<cv::Point2f>> imagePoints;    
vector<vector<int>> visibility;  //for each image, is each 3D point represented by a 2D image feature. 1 if yes, 0 if not.
vector<cv::Mat> cameraMatrix;  //The intrinsic matrix for each camera.
vector<cv::Mat> cameraRotations;
vector<cv::Mat> cameraTranslations;

// struct KeypointIndexesAndTriangul￼atedPoint {
//     int image1KeypointIndex;
//     int image2KeypointIndex;
//     cv::Point3f Point3DGuess;
// };

// struct LocalTransform {
//     cv::Mat rotation, translation;
// };

void fromCV2GLM(const cv::Mat& cvmat, glm::mat3* glmmat) {
    //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
    if (cvmat.cols != 3|| cvmat.rows != 3 || cvmat.type() != CV_32FC1) {
        cout << "Matrix conversion error! (3x3)" << endl;
        return;
    }
    memcpy(glm::value_ptr(*glmmat), cvmat.data, 12 * sizeof(float));
}

void fromCV2GLM(const cv::Mat& cvmat, glm::mat4* glmmat) {

    //TODO: This is temp.
    cv::Mat cvmat32;
    cvmat.convertTo(cvmat32, CV_32F);

    //Basic conversion method adapted from: https://stackoverflow.com/questions/44409443/how-a-cvmat-translate-from-to-a-glmmat4
    if (cvmat32.cols != 4|| cvmat32.rows != 4 || cvmat32.type() != CV_32FC1) {
        return;
    }
    memcpy(glm::value_ptr(*glmmat), cvmat32.data, 16 * sizeof(float));
}

void runSBA() {
        // if (imageSets.size() > 1) { //Only run if images > 2
            // run sba optimization
            // cout << "visibility " << endl;
            // for (vector<vector<int>>::const_iterator i = visibility.begin(); i != visibility.end(); ++i) {
            //     for (vector<int>::const_iterator j = (*i).begin(); j != (*i).end(); ++j) {
            //         cout << *j << ", ";
            //     }
            // cout << endl;
            // }
            
            // cout << "points3D " << endl;
            // for (vector<cv::Point3f>::const_iterator itr = points3D.begin(); itr != points3D.end(); ++itr) {
            //     cout << *itr << endl;
            // }

            // try {
                // sba.run( points3D, imagePoints, visibility, cameraMatrix, cameraRotations, cameraTranslations, distortionCoeffs);
            // } catch (cv::Exception) {

            // }

            //     cout << "relative translation after BA: " << imagePair->relativeTranslation << endl;
            //     cout << "image1->worldTranslation after BA" << imagePair->image1->worldTranslation << endl;
            //     cout << "image2->worldTranslation after BA" << imagePair->image2->worldTranslation << endl;

        // }
}

void setupSBA() {
    cvsba::Sba sba;
    cv::TermCriteria criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 150, 1e-10);
    cvsba::Sba::Params params;
    params.iterations = 150;
    params.type = cvsba::Sba::MOTIONSTRUCTURE;
    params.minError = 1e-10;
    params.fixedIntrinsics = 5;
    params.fixedDistortion = 5;
    params.verbose = false;
    sba.setParams(params);
}

void setupIntrinsicMatrix() {
        double cx = 5472/2;
        double cy = 3648/2;

        cv::Point2d pp(cx, cy);

        cameraIntrinsic = cv::Mat::eye(3, 3, CV_64F);

        cameraIntrinsic.at<double>(0,0) = FOCAL_LENGTH;
        cameraIntrinsic.at<double>(1,1) = FOCAL_LENGTH;
        cameraIntrinsic.at<double>(0,2) = cx;
        cameraIntrinsic.at<double>(1,2) = cy;
}

bool loadImagesAndDetectFeatures() {
    vector<filesystem::path> imagePaths;

    copy_if(filesystem::directory_iterator(imageDir), filesystem::directory_iterator(), back_inserter(imagePaths), [&](filesystem::path path){
        return find(acceptedExtensions.begin(), acceptedExtensions.end(), path.extension()) != acceptedExtensions.end();
    });
    sort(imagePaths.begin(), imagePaths.end());   //Sort, since directory iteration is not ordered on some file systems

    if (imagePaths.size() == 0) {
        cout << "No Images found at path!" << endl;
        return -1;
    } 

    for (vector<filesystem::path>::const_iterator itr = imagePaths.begin(); itr != imagePaths.end(); ++itr) {
        cv::String filePath = cv::String(filesystem::canonical(*itr).string()); //Get full file path, not relative.

        ImageData *currentImage = new ImageData(filePath, cameraIntrinsic,  cv::Mat::eye(3, 4, CV_64F));
        images.push_back(currentImage);
    }
    return 1;
}

cv::Mat estimateRelativeTransform(ImageData* image1, ImageData* image2, vector<cv::Point2f> image1Points, vector<cv::Point2f> image2Points) {
    cv::Point2d pp(cv::Mat(image1->cameraIntrinsic).at<double>(0,2), cv::Mat(image1->cameraIntrinsic).at<double>(1,2));
    double focal = cv::Mat(image1->cameraIntrinsic).at<double>(0,0);
    cv::Mat mask;

    cv::Mat essentialMat = cv::findEssentialMat(cv::Mat(image2Points), cv::Mat(image1Points), focal,
                                                    pp, cv::RANSAC, 0.999, 1.0, mask);

    //OpenCV returns a non 3x3 matrix if it can't derive an Essential Matrix.
    assert(essentialMat.cols == 3 && essentialMat.rows == 3);

    cv::Mat localTranslation, localRotation;
    cv::recoverPose(essentialMat, image2Points, image1Points, localRotation, localTranslation, focal, pp, mask);

    cv::Mat localTransform = cv::Mat::eye(4, 4, CV_64F);
    localRotation.copyTo(localTransform(cv::Range(0, 3), cv::Range(0, 3)));
    localTranslation.copyTo(localTransform(cv::Range(0, 3), cv::Range(3, 4)));

    return localTransform;
}

cv::Mat estimateWorldTransform(int image1Index, int image2Index, vector<cv::Point2f> image1Points, vector<cv::Point2f> image2Points) {
    cv::Mat localTransform = estimateRelativeTransform(images[image1Index], images[image2Index], image1Points, image2Points);
    images[image2Index]->worldTransform = images[image1Index]->worldTransform * localTransform;
}

vector<cv::Point3f> triangulatePoints(ImageData* image1, ImageData* image2, vector<cv::Point2f> image1Points, vector<cv::Point2f> image2Points) {
    //TODO: Rewrite this to conform to the other code, and so that it uses image.transform rather than seperate 

    cv::Mat i1WorldTransformation = cv::Mat::eye(3, 4, CV_64FC1); 
    image1->worldTransform.rowRange(0, 3).colRange(0, 4).copyTo(i1WorldTransformation);

    cv::Mat i2WorldTransformation = cv::Mat::eye(3, 4, CV_64FC1);
    image2->worldTransform.rowRange(0, 3).colRange(0, 4).copyTo(i2WorldTransformation);

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

void matchFeatures(int image1Index, int image2Index) {
    ImageData* image1 = images[image1Index]; 
    ImageData* image2 = images[image2Index];

    //Feature Match 
        // cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");        
        // Match features between all images
        vector<cv::Point2f> initialMatchesP1, initialMatchesP2,  filteredMatchesP1, filteredMatchesP2;;
        vector<int> intialIndexesP1, intialIndexesP2, filteredIndexesP1, filteredIndexesP2;
        std::vector<std::vector<cv::DMatch>> knn_matches;
        std::vector<cv::DMatch> initialMatches, filteredMatches;
        matcher->knnMatch((*image1).image_descriptors, (*image2).image_descriptors, knn_matches, 2);

    //Lowes Ratio Test Filter
        const float ratio_thresh = 0.7f;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() > 0 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                initialMatchesP1.push_back(image1->image_keypoints[knn_matches[i][0].queryIdx].pt);
                initialMatchesP2.push_back(image2->image_keypoints[knn_matches[i][0].trainIdx].pt);

                //Store the indexes to avoid the problem of having keypoints not matching up when masked.
                intialIndexesP1.push_back(knn_matches[i][0].queryIdx);
                intialIndexesP2.push_back(knn_matches[i][0].trainIdx);

                initialMatches.push_back(knn_matches[i][0]);
            }
        }

    //Fundamental Matrix Constraint Filter
        vector<uchar> mask;
        cv::findFundamentalMat(initialMatchesP1, initialMatchesP2, cv::FM_RANSAC, 3.0, 0.99, mask);
        
        for (size_t k=0; k < mask.size(); k++) {
            if (mask[k]) {
                filteredIndexesP1.push_back(intialIndexesP1[k]);
                filteredIndexesP2.push_back(intialIndexesP2[k]);

                filteredMatchesP1.push_back(image1->image_keypoints[intialIndexesP1[k]].pt);
                filteredMatchesP2.push_back(image2->image_keypoints[intialIndexesP2[k]].pt);

                filteredMatches.push_back(initialMatches.at(k));
            }
        }


        cv::Mat img_matches;
        cv::drawMatches(image1->image, image1->image_keypoints, image2->image, image2->image_keypoints,
            filteredMatches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
            vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        //Show detected matches in an image viewer for debug purposes. 
        cv::imshow("Good Matches", img_matches);
        cv::waitKey(0); //Wait for a key to be hit to exit viewer.


    //Initial estimate for World Position of Image2
        cv::Mat localTransform = estimateRelativeTransform(images[image1Index], images[image2Index], filteredMatchesP1, filteredMatchesP2);
        images[image2Index]->worldTransform = images[image1Index]->worldTransform * localTransform;

    //Triangulate initial guesses for Image2
        vector<cv::Point3f> currentPair3DGuesses = triangulatePoints(image1, image2, filteredMatchesP1, filteredMatchesP2);
        int good_matches = cv::sum(mask)[0];
        assert(good_matches >= 10);


    //Calculate scale factor based on previous points
        cv::Point3f previousPairGuess1, previousPairGuess2, currentPairGuess1, currentPairGuess2; 
        bool firstPair = true;
        double previousPairScale = 0, currentPairScale = 0;
        int count;
        for (int i = 0; i < currentPair3DGuesses.size(); i++) {
            auto corresponding3DPoint = previousPairImage2FeaturesToPoints3D.find(filteredMatchesP1[i]);
            if (corresponding3DPoint != previousPairImage2FeaturesToPoints3D.end()) {
                if(firstPair) {
                    //The first time we've found a set of three matched points, set them to guess 1 rather than 2.
                    previousPairGuess1 = points3DGuesses[corresponding3DPoint->second].back();
                    currentPairGuess1 = currentPair3DGuesses[i];
                    firstPair = false;
                } else {
                    //We have two points matched across each pair, we can do scaling.
                    previousPairGuess2 = points3DGuesses[corresponding3DPoint->second].back();
                    currentPairGuess2 = currentPair3DGuesses[i];

                    previousPairScale += cv::norm(cv::Mat(previousPairGuess1) - cv::Mat(previousPairGuess2));
                    currentPairScale += cv::norm(cv::Mat(currentPairGuess1) - cv::Mat(currentPairGuess2));
                    count++;

                    previousPairGuess1 = previousPairGuess2;
                    currentPairGuess1 = currentPairGuess2;
                }
            }
        }

        if (previousPairScale != 0 && currentPairScale != 0) {
            //Scale
            double scaleFactor = (previousPairScale / currentPairScale) / count;

            cout << "previous Scale: " << previousPairScale 
                 << ", current Scale: " << currentPairScale 
                 << "\nScaleFactor: " << scaleFactor << endl;

            //Adjust estimate for World Position of Image2
            localTransform *= scaleFactor;
            images[image2Index]->worldTransform = images[image1Index]->worldTransform * localTransform;

            //Retriangulate Points
                currentPair3DGuesses = triangulatePoints(image1, image2, filteredMatchesP1, filteredMatchesP2);
                int good_matches = cv::sum(mask)[0];
                assert(good_matches >= 10);
        }


    // Put points into final structure.
    map<cv::Point2f, int> currentPairImage2FeaturesToPoints3D;
    for (int i = 0; i < currentPair3DGuesses.size(); i++) {
        auto corresponding3DPoint = previousPairImage2FeaturesToPoints3D.find(filteredMatchesP1[i]);

        if (corresponding3DPoint != previousPairImage2FeaturesToPoints3D.end()) {
            // int point3DGuessIndex = std::distance(points3DGuesses.begin(), points3DGuesses[corresponding3DPoint->second].back);
            points3DGuesses[corresponding3DPoint->second].push_back(currentPair3DGuesses[i]);

            //Create a binding from the image2 point to the index of the 3D guess list.                
            currentPairImage2FeaturesToPoints3D[filteredMatchesP2[i]] = corresponding3DPoint->second;
        } else { //New point
            //Create a new list of 3D point guesses if we're on a new point.
            vector<cv::Point3f> newGuessList;
            newGuessList.push_back(currentPair3DGuesses[i]);
            points3DGuesses.push_back(newGuessList);

            //Create a binding from the image2 point to the index of the 3D guess list.
            int new3DGuessindex =  points3DGuesses.size()-1;
            currentPairImage2FeaturesToPoints3D[filteredMatchesP2[i]] = new3DGuessindex;
        }   
    }

    previousPairImage2FeaturesToPoints3D = currentPairImage2FeaturesToPoints3D;
    currentPairImage2FeaturesToPoints3D.clear();
}

int main(int argc, const char* argv[]) {
    cout << "Launching Program" << endl;

	srand (time(NULL));
    setupIntrinsicMatrix();
    setupSBA();

    if (!loadImagesAndDetectFeatures()) return -1;

    // Match features between all images
    for (int i = 0; i < images.size() - 1; i++) {
        // for (int j=i+1; j < images.size(); j++) {
            matchFeatures(i, i+1);
        // }
    }

    for (int i = 0; i < points3DGuesses.size(); i++) {
        vector<cv::Point3f> currentPointGuesses = points3DGuesses[i];
        cv::Point3f averagedPoint;
        if (currentPointGuesses.size() > 3) {
            for (int j = 0; j < currentPointGuesses.size(); j++) {
                averagedPoint += currentPointGuesses[j];
            }
            averagedPoint /= ((float) currentPointGuesses.size());
            cout << averagedPoint << endl;
            // averagedPoint *= 10.0f;
            points3D.push_back(glm::vec3(averagedPoint.x, averagedPoint.y, averagedPoint.z));
        }
    }

    renderEnvironment *renderer = new renderEnvironment();

    cout << "Initialised renderer" << endl;
        	
    GLuint basicShader = Shader::LoadShaders("./bin/shaders/basic.vertshader", "./bin/shaders/basic.fragshader");
	// renderer->addRenderable(new Renderable(basicShader, cameraPosesToRender, cameraColoursToRender, GL_POINTS));
	renderer->addRenderable(new Renderable(basicShader, points3D, points3D, GL_POINTS));

    while (true) {  //TODO: Write proper update & exit logic.
		oldTime = newTime;
    	newTime = chrono::steady_clock::now();
		deltaT = chrono::duration_cast<chrono::milliseconds>(newTime - oldTime).count();

        renderer->update(deltaT);
    }
    return 0;
}






#include <iostream>
#include <chrono>
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

const string imageDir = "./bin/data/dinoRing/";
vector<string> acceptedExtensions = {".png", ".jpg", ".PNG", ".JPG"};

const double FOCAL_LENGTH = 3310.400000; //focal length in pixels, after downsampling, guess from jpeg EXIF data

//Given with Dataset.
const cv::Matx33d cameraIntrinsic (3310.400000f, 0.000000f, 320.0000f,
                                0.000000f, 3325.500000f, 240.0000f,
                                0.000000f, 0.000000f, 1.000000f);

double initPose[]{-0.08661715685291285200, 0.97203145042392447000, 0.21829465483805316000, -0.97597093004059532000, -0.03881511324024737600,
    -0.21441803766270939000, -0.19994795321325870000, -0.23162091782017200000, 0.95203517502501223000, -0.0526034704197, 0.023290917003, 0.659119498846};

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
vector<glm::vec3> cameraPosesToRender, cameraColoursToRender;
vector<cv::Point3f> points3D;    //3D Points
// vector<vector<cv::Point2f> >  imagePoints;    //List of a list of each images detected features
// vector<vector<int> > visibility;  //for each image, is each 3D 
vector<cv::Mat> cameraMatrix;  //The intrinsic matrix for each camera.
vector<cv::Mat> cameraRotations;
vector<cv::Mat> cameraTranslations;
vector<cv::Mat> distortionCoeffs;

struct KeypointIndexesAndTriangulatedPoint {
    int image1KeypointIndex;
    int image2KeypointIndex;
    cv::Point3f Point3DGuess;
};

struct LocalTransform {
    cv::Mat rotation, translation;
};

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

        ImageData *currentImage = new ImageData(filePath, cameraIntrinsic,  cv::Mat::zeros(3, 4, CV_64F));
        images.push_back(currentImage);
    }
    return 1;
}

LocalTransform EstimateRelativePose(ImageData* image1, ImageData* image2, vector<cv::Point2f> image1Points, vector<cv::Point2f> image2Points) {
    cv::Point2d pp(cv::Mat(image1->cameraIntrinsic).at<double>(0,2), cv::Mat(image1->cameraIntrinsic).at<double>(1,2));
    double focal = cv::Mat(image1->cameraIntrinsic).at<double>(0,0);
    cv::Mat mask;

    cv::Mat essentialMat = cv::findEssentialMat(cv::Mat(image2Points), cv::Mat(image1Points), focal,
                                                    pp, cv::RANSAC, 0.999, 1.0, mask);

    //OpenCV returns a non 3x3 matrix if it can't derive an Essential Matrix.
    assert(essentialMat.cols == 3 && essentialMat.rows == 3);

    LocalTransform* result = new LocalTransform;
    cv::recoverPose(essentialMat, image2Points, image1Points, result->rotation, result->translation, focal, pp, mask);

    return *result;
}

cv::Mat EstimateWorldPose(int image1Index, int image2Index, vector<cv::Point2f> image1Points, vector<cv::Point2f> image2Points) {
    LocalTransform image1ToImage2 = EstimateRelativePose(images[image1Index], images[image2Index], image1Points, image2Points);

    cv::Mat image2WorldTranslation, image2WorldRotation;

    image2WorldTranslation = images[image1Index]->worldTranslation * image1ToImage2.translation;
    image2WorldTranslation = images[image1Index]->worldRotation * image1ToImage2.translation;
    if (image1Index == 0) {
        //First image pair in the set; no need to do any scaling.
        images[image2Index]->worldTranslation = 

    } else {
        images[image1Index-1]->worldTranslation
    }
}

vector<cv::Point3f> TriangulatePoints(ImageData* image1, ImageData* image2, vector<cv::Point2f> image1Points, vector<cv::Point2f> image2Points) {
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

void matchFeatures(ImageData* image1, ImageData* image2) {
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    // Match features between all images
    vector<cv::Point2f> initialMatchesP1, initialMatchesP2;
    vector<int> intialIndexesP1, intialIndexesP2;
    vector<int> filteredIndexesP1, filteredIndexesP2;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch((*image1).image_descriptors, (*image2).image_descriptors, knn_matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() > 0 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {  
            initialMatchesP1.push_back(image1->image_keypoints[knn_matches[i][0].queryIdx].pt);
            initialMatchesP2.push_back(image2->image_keypoints[knn_matches[i][0].trainIdx].pt);

            //Store the indexes to avoid the problem of having keypoints not matching up when masked.
            intialIndexesP1.push_back(knn_matches[i][0].queryIdx);
            intialIndexesP2.push_back(knn_matches[i][0].trainIdx);
        }
    }

    // Filter bad matches using fundamental matrix constraint
    vector<KeypointIndexesAndTriangulatedPoint> triangulatedPoints;

    vector<uchar> mask;
    cv::findFundamentalMat(initialMatchesP1, initialMatchesP2, cv::FM_RANSAC, 3.0, 0.99, mask);

    //Calculate World Position of Image2

    //Triangulate initial guesses for Image2
    vector<cv::Point3f> points3D = TriangulatePoints(image1, image2, initialMatchesP1, initialMatchesP2);

    for (size_t k=0; k < mask.size(); k++) {
        if (mask[k]) {
            KeypointIndexesAndTriangulatedPoint match;
            // img_pose_i.kp_match_idx(i_kp[k], j) = j_kp[k];
            // img_pose_j.kp_match_idx(j_kp[k], i) = i_kp[k];

            match.image1KeypointIndex = intialIndexesP1[k];
            match.image2KeypointIndex = intialIndexesP2[k];
            match.Point3DGuess = points3D[k];
            triangulatedPoints.push_back(match);
        }
    }
    int good_matches = cv::sum(mask)[0];
    assert(good_matches >= 10);
}

int main(int argc, const char* argv[]) {
    cout << "Launching Program" << endl;

	srand (time(NULL));
    setupSBA();

    if (!loadImagesAndDetectFeatures()) return -1;

    // Match features between all images
    for (int i = 0; i < images.size() - 1; i++) {
        ImageData* image1 = images[i];
        for (size_t j=i+1; j < images.size(); j++) {
            ImageData* image2 = images[j];
            matchFeatures(image1, image2);


        }
    }












    // //Load initial image and remove it from queue.
    // cout << "creating first imagedata" << endl;
    // ImageData *previousImage = new ImageData(cv::String(filesystem::canonical(imagePaths[0]).string()), cameraIntrinsic, initialPose);
    // imagePaths.erase(imagePaths.begin());

    // vector<cv::Point3f> prevPoints;
    
    // //Create image pairs.
    // for (vector<filesystem::path>::const_iterator itr = imagePaths.begin(); itr != imagePaths.end(); ++itr) {
    //     cv::String filePath = cv::String(filesystem::canonical(*itr).string()); //Get full file path, not relative.

    //     ImageData *currentImage = new ImageData(filePath, cameraIntrinsic,  cv::Mat::zeros(3, 4, CV_64F));
    //     ImageDataSet *imagePair = new ImageDataSet(previousImage, currentImage);
    //     imageSets.push_back(imagePair);

    //     vector<cv::Point3f> newPoints;

    //     imagePoints.push_back(imagePair->points1); //Push back image 1's points. Image 2's points will be pushed back as next iterations' points1.

    //     if (imageSets.size() == 1) {    //If it's the first image pair, all 3D points are new!
    //         newPoints = imagePair->TriangulatePoints(imagePair->points1, imagePair->points2);
    //         points3D.insert(points3D.end(), newPoints.begin(), newPoints.end());
                        
    //         cameraMatrix.push_back(cv::Mat(cameraIntrinsic));    //Camera 1
    //         cameraRotations.push_back(cv::Mat(previousImage->worldRotation));
    //         cameraTranslations.push_back(cv::Mat(previousImage->worldTranslation));
    //         distortionCoeffs.push_back(cv::Mat::zeros(5, 1, CV_64F));
            
    //         vector<int> cameraVisibilities;
    //         cout << "Found N matches: " << newPoints.size() << endl;
    //         for (int i = 0; i < newPoints.size(); i++) {
    //             cameraVisibilities.push_back(1);    //All points are visible to both cameras.
    //             //TODO: This might be wrong. We're assuming triangulated points are returned in the right order..
    //             imageSets[imageSets.size()-1]->visibilityLocations[imageSets[imageSets.size()-1]->points2[i]] = i; 
    //         }

    //         visibility.push_back(cameraVisibilities); //Camera 1
    //         visibility.push_back(cameraVisibilities); //Camera 2

    //         //Setup for next iteration. this is normally copied from points.
    //         prevPoints = imageSets[imageSets.size()-1]->TriangulatePoints(imageSets[imageSets.size()-1]->points1, imageSets[imageSets.size()-1]->points2); 
    //     } else {
    //         vector<int> cameraVisibilities;
    //         //Start by adding the new camera to the end of the camera list in visibility, initialised to all points not visible in this image (0)
    //         for (int i = 0; i < visibility[0].size(); i++) {
    //             cameraVisibilities.push_back(0);
    //         }
    //         visibility.push_back(cameraVisibilities);   //Add new line to the array.

    //         vector<cv::Point3f> pair1MatchedPoints, pair2MatchedPoints;

    //         //Triangulate points
    //         //Todo: program flow needs to be adjusted so we're not redoing this all the time?
    //         vector<cv::Point3f> prevPoints = imageSets[imageSets.size()-2]->TriangulatePoints(imageSets[imageSets.size()-2]->points1, imageSets[imageSets.size()-2]->points2); 
    //         vector<cv::Point3f> points = imagePair->TriangulatePoints(imagePair->points1, imagePair->points2);
            
    //         int matches = 0;
    //         for (int i = 0; i < imagePair->points1.size(); i++) {

    //             cv::Point2f image1Point = imagePair->points1[i], image2Point = imagePair->points2[i];

    //             //image1Point != previousPair[image2Point]. This is because points1/2 is not the same between pairs, as it's only GOOD matches for the pair,
    //             //...not all of the possible matches.
    //             //prevPairPoints2[prevPairPoint2Index] == image1Point
    //             int prevPairPoint2Index;
    //             vector<cv::Point2f> prevPairPoints2 = imageSets[imageSets.size()-2]->points2;
    //             auto it = std::find(prevPairPoints2.begin(), prevPairPoints2.end(), image1Point);
    //             if (it != prevPairPoints2.end()) {
    //                 prevPairPoint2Index = std::distance(prevPairPoints2.begin(), it);
    //             } else {
    //                 continue;
    //             }

    //             std::map<cv::Point2f, int>::iterator visibilityLocation = imageSets[imageSets.size()-2]->visibilityLocations.find(image1Point);
    //             // cout << "Trying to find a match for: " << image1Point << endl; 
    //             if (visibilityLocation != imageSets[imageSets.size()-2]->visibilityLocations.end()) {
    //                 matches++;

    //                 //If the point exists in a previous imageset, then the 3D point has already been added to the list and we should
    //                 //...append to that visibility list rather than making a new one.
    //                 imageSets[imageSets.size()-1]->visibilityLocations[image2Point] = i;
    //                 visibility[visibility.size() - 1][visibilityLocation->second] = 1;

    //                 pair1MatchedPoints.push_back(prevPoints[prevPairPoint2Index]);  
    //                 pair2MatchedPoints.push_back(points[i]);  

    //                 cout << "New Pair of pairs: \n" << prevPairPoints2[prevPairPoint2Index] << endl << imagePair->points1[i] << endl;
    //                 cout << "previous Point: " << prevPoints[prevPairPoint2Index] << endl;
    //                 cout << "current Point: " << points[i] << endl;

    //             } else { //New point, Only visible in the most recent image pair.
    //                 // cout << "No Match Found" << endl;

    //                 vector<int> cameraVisibilities;         
    //                 for (int i = 0; i < visibility.size()-2; i++) { 
    //                     visibility[i].push_back(0); 
    //                 }

    //                 visibility[visibility.size()-2].push_back(1);  //Camera 1
    //                 visibility[visibility.size()-1].push_back(1);  //Camera 2
    //             }
    //         }

    //         // recalculate relative transform with scaled local transform
    //         if (pair1MatchedPoints.size() > 0) {
    //             double i1Scale = 0, i2Scale = 0;

    //             for (int i = 1; i < pair1MatchedPoints.size(); i++) {
    //                 i1Scale += cv::norm(cv::Mat(pair1MatchedPoints[i-1]) - cv::Mat(pair1MatchedPoints[i]));
    //                 i2Scale += cv::norm(cv::Mat(pair2MatchedPoints[i-1]) - cv::Mat(pair2MatchedPoints[i]));
    //             }

    //             double relativeScale = i1Scale/i2Scale;
    //             cout << "image1->worldTranslation before" << imagePair->image1->worldTranslation << endl;
    //             cout << "image2->worldTranslation before" << imagePair->image2->worldTranslation << endl;
    //             cout << "i1Scale: " << i1Scale << ", i2Scale: " << i2Scale << ", relativeScale: " << relativeScale << endl;
    //             cout << "relative translation before: " << imagePair->relativeTranslation << endl;

    //             imagePair->relativeTranslation /= relativeScale;

    //             // Recalculate projection matrix
    //             // Construct a transformation mat from a translation and a rotation mat.
    //             cv::Mat i1WorldTransformation = cv::Mat::eye(3, 4, CV_64F),
    //                         relativeTransformation = cv::Mat::eye(3, 4, CV_64F);       
    //             previousImage->worldRotation.copyTo(i1WorldTransformation.rowRange(0,3).colRange(0,3));
    //             previousImage->worldTranslation.copyTo(i1WorldTransformation.rowRange(0,3).col(3));

    //             imagePair->relativeRotation.copyTo(relativeTransformation.rowRange(0,3).colRange(0,3));
    //             imagePair->relativeTranslation.copyTo(relativeTransformation.rowRange(0,3).col(3));

    //             //Multiply the two transforms
    //             cv::Mat result = cv::Mat::eye(3, 4, CV_64F);        
    //             cv::multiply(i1WorldTransformation, relativeTransformation, result);

    //             //Split back into separate rotations/translations.
    //             currentImage->worldRotation = result.rowRange(0,3).colRange(0,3);
    //             currentImage->worldTranslation = result.rowRange(0,3).col(3);

    //             cout << "relative translation after: " << imagePair->relativeTranslation << endl;
    //             cout << "image1->worldTranslation after" << imagePair->image1->worldTranslation << endl;
    //             cout << "image2->worldTranslation after" << imagePair->image2->worldTranslation << endl;
    //         }

    //         //Retriangulate points
    //         newPoints = imagePair->TriangulatePoints(imagePair->points1, imagePair->points2);
    //         points3D.insert(points3D.end(), newPoints.begin(), newPoints.end());
    //     }

    //     //DEBUG
    //     glm::mat4 glmPose;
    //     cv::Mat cvPose = cv::Mat::eye(4, 4, CV_64F);
    //     currentImage->worldRotation.copyTo(cvPose.rowRange(0,3).colRange(0,3));
    //     currentImage->worldTranslation.copyTo(cvPose.rowRange(0,3).col(3));

    //     fromCV2GLM(cvPose, &glmPose);
    //     glm::vec3 cameraPos = glm::vec3(glm::vec4(10.0) * glmPose).xyz;

    //     // cout << "imageSets.size(): " << imageSets.size() << endl;
    //     // cout << "Vector 3 Transformation: " << glm::to_string(cameraPos) << endl;
    //     // cout << "Translation: " << currentImage->worldTranslation << endl;
    //     // cout << "Rotation: " << currentImage->worldRotation << endl << endl;

    //     cameraMatrix.push_back(cv::Mat(cameraIntrinsic));    //Camera 2
    //     cameraRotations.push_back(currentImage->worldRotation);
    //     cameraTranslations.push_back(currentImage->worldTranslation);
    //     distortionCoeffs.push_back(cv::Mat::zeros(5, 1, CV_64F));

    //     previousImage = currentImage;

    //     runSBA();
    // }

    // for (int i = 0; i < cameraRotations.size() ; i++) {
    //     glm::mat4 glmPose;
    //     cv::Mat cvPose = cv::Mat::eye(4, 4, CV_64F); 
    //     cameraRotations[i].copyTo(cvPose.rowRange(0,3).colRange(0,3));
    //     cameraTranslations[i].copyTo(cvPose.rowRange(0,3).col(3));

    //     fromCV2GLM(cvPose, &glmPose);

    //     glm::vec3 cameraPos = glm::vec3(glm::vec4(1.0) * glmPose).xyz;
    //     cout << "Camera Pose: " << glm::to_string(cameraPos) << endl;
    //     cameraPosesToRender.push_back(cameraPos);
    //     cameraColoursToRender.push_back(glm::vec3(  ((((double) rand() / (RAND_MAX))/0.7)+0.3), 
    //                                                 ((((double) rand() / (RAND_MAX))/0.7)+0.3), 
    //                                                 ((((double) rand() / (RAND_MAX))/0.7)+0.3)
    //                                             ));
    // }

    // vector<glm::vec3> pointsToRender;
    // for (int i = 0; i < points3D.size() ; i++) {
    //     glm::vec3 glmPoint(points3D[i].x, points3D[i].y, points3D[i].z);
    //     // cout << glm::to_string(glmPoint) << endl;
    //     pointsToRender.push_back(glmPoint);
    // }
    
    renderEnvironment *renderer = new renderEnvironment();
    cout << "Initialised renderer" << endl;
        	
    GLuint basicShader = Shader::LoadShaders("./bin/shaders/basic.vertshader", "./bin/shaders/basic.fragshader");
	renderer->addRenderable(new Renderable(basicShader, cameraPosesToRender, cameraColoursToRender, GL_POINTS));
	// renderer->addRenderable(new Renderable(basicShader, pointsToRender, pointsToRender, GL_POINTS));

    while (true) {  //TODO: Write proper update & exit logic.
		oldTime = newTime;
    	newTime = chrono::steady_clock::now();
		deltaT = chrono::duration_cast<chrono::milliseconds>(newTime - oldTime).count();

        renderer->update(deltaT);
    }
    return 0;
}






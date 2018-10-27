#include <iostream>
#include <chrono>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <GL/glew.h>
#define GLM_SWIZZLE
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

#include <cvsba/cvsba.h>

using namespace std;
using namespace boost;

const string imageDir = "./bin/data/dinoRing/";

//Given with Dataset.
const cv::Matx33f cameraIntrinsic (3310.400000f, 0.000000f, 316.730000f,
                                0.000000f, 3325.500000f, 200.550000f,
                                0.000000f, 0.000000f, 1.000000f);

float initPose[]{-0.08661715685291285200, 0.97203145042392447000, 0.21829465483805316000, -0.97597093004059532000, -0.03881511324024737600,
    -0.21441803766270939000, -0.19994795321325870000, -0.23162091782017200000, 0.95203517502501223000, -0.0526034704197, 0.023290917003, 0.659119498846};

//We'd normally assume world origin = intial camera position, but we're given it in the dino dataset, we use it here so we can check other camera poses.
const cv::Matx34f initialPose = cv::Mat(3, 4, CV_32F, initPose);
 


vector<ImageDataSet*> imageSets;
vector<string> acceptedExtensions = {".png", ".jpg"};

vector<vec3> test_data_lines = {
	vec3(0.000000, 4.000000, 12.000000),
	vec3(2.870316, -101.411970, 1.000000),

	vec3(0.000000, 0.000000, 0.2233231),
	vec3(3.897837, 14.548908, 1.000000),

	vec3(4.398729, 18.116240, 2.213213),
	vec3(5.055743, 6.247947, 1.000000),

	vec3(5.365756, 6.035206, 4.000000),
	vec3(5.693254, 14.517317, 1.00000)
};

auto oldTime = chrono::steady_clock::now(), newTime = chrono::steady_clock::now();
double deltaT;

vector<glm::vec3> cameraPosesToRender;
vector<cv::Point3f> points3D;    //3D Points
vector<vector<cv::Point2f> >  imagePoints;    //List of a list of each images points
vector<vector<int> > visibility;  //for each image, is each 3D 
vector<cv::Mat> cameraMatrix;  //The intrinsic matrix for each camera.
vector<cv::Mat> cameraRotations;
vector<cv::Mat> cameraTranslations;
vector<cv::Mat> distortionCoeffs;

int main(int argc, const char* argv[]) {
    cout << initialPose << endl;
    cout << "Launching Program" << endl;
	srand (time(NULL));

    if (!filesystem::exists(imageDir) || filesystem::is_empty(imageDir)) {
        cout << "No Images found at path!" << endl;
        return -1;
    }

    vector<filesystem::path> v;
    copy_if(filesystem::directory_iterator(imageDir), filesystem::directory_iterator(), back_inserter(v), [&](filesystem::path path){
        return find(acceptedExtensions.begin(), acceptedExtensions.end(), path.extension()) != acceptedExtensions.end();
    });
    sort(v.begin(), v.end());   //Sort, since directory iteration is not ordered on some file systems

    glm::vec3 colour = *new glm::vec3(1.0f, 0.2f, 0.0f);

    //Load initial image and remove it from queue.
    cout << "creating first imagedata" << endl;
    ImageData *previousImage = new ImageData(cv::String(filesystem::canonical(v[0]).string()), cameraIntrinsic,
                                colour, initialPose);
    v.erase(v.begin());
    cout << "First imagedata created" << endl;

    //TODO: run through method to convert to glm.
    // glm::vec3 cameraPos = (vec4(1.0, 1.0, 1.0, 0.0) * previousImage->worldTransformation).xyz();
    // cameraPosesToRender.push_back(cameraPos);
    
    //Create image pairs.
    for (vector<filesystem::path>::const_iterator itr = v.begin(); itr != v.end(); ++itr) {
        cv::String filePath = cv::String(filesystem::canonical(*itr).string()); //Get full file path, not relative.

        ImageData *currentImage = new ImageData(filePath, cameraIntrinsic, colour, cv::Matx34f::zeros());
        ImageDataSet *imagePair = new ImageDataSet(previousImage, currentImage);
        imageSets.push_back(imagePair);

        //TODO: run through method to convert to glm.
        // cameraPos = (cv::Mat(1.0, 1.0, 1.0, 0.0) * cv::Mat(currentImage->worldTransformation)).;
        // cameraPosesToRender.push_back(cameraPos);

        vector<cv::Point3f> newPoints;

        // vector<cv::Point3f> points3D;    //3D Points
        // vector<vector<cv::Point2f> >  imagePoints;    //List of a list of each images points
        // vector<vector<int> > visibility;  //for each image, is each 3D 
        // vector<cv::Mat> cameraMatrix;  //The intrinsic matrix for each camera.
        // vector<cv::Mat> cameraRotations;
        // vector<cv::Mat> cameraTranslations;
        // vector<cv::Mat> distortionCoeffs;

        /*
        create new image_data_set, detecting points
        for each key feature in the new image
            add to imagePoints
            if it exists in any previous(?) image pair, it's part of a point track
                add to point track
            else
                triangulate point 
                add point
        */

        //Push back image 1's points. Image 2's points will be pushed back as next iterations' points1.
        imagePoints.push_back(imagePair->points1); 

        if (imageSets.size() == 1) {    //If it's the first image pair, all 3D points are new!
            cout << "imageSets.size() = 1" << endl;
            newPoints = imagePair->TriangulatePoints(imagePair->points1, imagePair->points2);
            points3D.insert(points3D.end(), newPoints.begin(), newPoints.end());
            
            for (int i = 0; i < newPoints.size(); i++) {
                vector<int> cameraVisibilities;
                cameraVisibilities.push_back(1); //Camera 1
                cameraVisibilities.push_back(1); //Camera 2
                visibility.push_back(cameraVisibilities);
                imageSets[imageSets.size()-1]->visibilityLocations[imageSets[imageSets.size()-1]->points2[i]] = i; //TODO: This might be wrong. We're assuming triangulated points are returned in the right order..
            }
        } else {

            //Start by adding the new camera to the end of each point list in visibility, initialised to point not visible in this image (0)
            for (vector<vector<int>>::iterator pointList = visibility.begin(); pointList != visibility.end(); ++pointList) {
                (*pointList).push_back(0);
            }

            for (vector<cv::Point2f>::iterator point = imagePair->points1.begin(); point != imagePair->points1.end(); ++point) {
                std::map<cv::Point2f, int>::iterator visibilityLocation = imageSets[imageSets.size()-2]->visibilityLocations.find(*point);

                // cout << "visibilityLocations for last pair:" << endl;
                // for(map<cv::Point2f, int>::const_iterator it = imageSets[imageSets.size()-2]->visibilityLocations.begin();
                //     it != imageSets[imageSets.size()-2]->visibilityLocations.end(); ++it)
                // {
                //     std::cout << it->first << endl;
                // }

                cout << "Trying to find a match for: " << *point << endl; 
                if (visibilityLocation != imageSets[imageSets.size()-2]->visibilityLocations.end()) {

                    cout << "Found a match" << endl;
                    //If the point exists in a previous imageset, then the 3D point has already been added to the list and we should
                    //...append to that visibility list rather than making a new one.
                    imageSets[imageSets.size()-1]->visibilityLocations[*point] = visibilityLocation->second;
                    visibility[visibilityLocation->second][visibility[visibilityLocation->second].size() - 1] = 1;
                } else { //New point, Only visible in the most recent image pair.
                    cout << "No Match Found" << endl;

                    vector<int> cameraVisibilities;        
                    for (int i = 0; i < imageSets.size()-1; i++) 
                    { 
                        cameraVisibilities.push_back(0); 
                    }
                    cameraVisibilities.push_back(1);  //Camera 1
                    cameraVisibilities.push_back(1);  //Camera 2 
                    visibility.push_back(cameraVisibilities);
                }
            }
        }
        cameraMatrix.push_back(cv::Mat(cameraIntrinsic));    //Camera 1
        // cameraTranslations.push_back(imageSets[imageSets.size()-1]->image1->worldTransformation.

        cameraMatrix.push_back(cv::Mat(cameraIntrinsic));    //Camera 2 

        previousImage = currentImage;

        // cout << "visibility " << endl;
        // for (vector<vector<int>>::const_iterator i = visibility.begin(); i != visibility.end(); ++i) {
        //     for (vector<int>::const_iterator j = (*i).begin(); j != (*i).end(); ++j) {
        //         cout << *j << ", ";
        //     }
        // cout << endl;
        // }
    }

    // cout << "points3D " << endl;
    // for (vector<cv::Point3f>::const_iterator itr = points3D.begin(); itr != points3D.end(); ++itr) {
    //     cout << *itr << endl;
    // }

    // cout << "imagePoints " << endl;
    // for (vector<vector<cv::Point2f>>::const_iterator i = imagePoints.begin(); i != imagePoints.end(); ++i) {
    //     for (vector<cv::Point2f>::const_iterator j = (*i).begin(); j != (*i).end(); ++j) {
    //         cout << *j << ", ";
    //     }
    //     cout << endl;
    // }

    // cout << "Camera Positions: " << endl;
    // for (vector<glm::vec3>::const_iterator itr = cameraPosesToRender.begin(); itr != cameraPosesToRender.end(); ++itr) {
    //     cout << glm::to_string(*itr) << endl;
    // }

    // cout << "Camera Poses: " << endl;
    // cout << glm::to_string(imageSets[0]->image1->worldTransformation) << endl;
    // for (vector<ImageDataSet*>::const_iterator itr = imageSets.begin(); itr != imageSets.end(); ++itr) {
    //     cout << glm::to_string((*itr)->image2->worldTransformation) << endl;
    // }

    //  * @param  cameraMatrix M x 3 x 3 camera matrix (intrinsic parameters) 3 x 3 camera matrix for each image
    //  * @param  distCoeffs M x   5  x1  distortion coefficient  for each image
    //  * @param R  M x 3 x 3 rotation matrix  for each image
    //  * @param T M x 3 x 1 translation matrix  for each image




//    //Point triangulation and Bundle Adjustment.
//     for (vector<ImageDataSet*>::const_iterator itr = imageSets.begin(); itr != imageSets.end(); ++itr) {
//         if (itr = imageSets.begin) {

//         } else {
//             // create new image_data_set, detecting points
//             //     for each key feature in the new image
//             //         add a new line to cameraMatrix
//             //         add to imagePoints
//             //         if it exists in the last image pair, it's part of a point track
//             //             push back a zero for all cameras in visibility
//             //             go through the point track, setting each cameras visibility vectors last added element to 1
//             //         else
//             //             triangulate point 
//             //             add triangulated point to points3D
//         }
//     }



    renderEnvironment *renderer = new renderEnvironment();
    // cout << "Initialised renderer" << endl;
        	
    // GLuint basicShader = Shader::LoadShaders("./bin/shaders/basic.vertshader", "./bin/shaders/basic.fragshader");
	// renderer->addRenderable(new Renderable(basicShader, cameraPosesToRender, cameraPosesToRender, GL_POINTS));

    // while (true) {  //TODO: Write proper update & exit logic.
	// 	oldTime = newTime;
    // 	newTime = chrono::steady_clock::now();
	// 	deltaT = chrono::duration_cast<chrono::milliseconds>(newTime - oldTime).count();

    //     renderer->update(deltaT);
    // }
    return 0;
}


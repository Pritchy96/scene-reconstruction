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

#include "../include/image_data.hpp"
#include "../include/image_data_set.hpp"
#include "../include/render_environment.hpp"
#include "../include/shader.hpp"

#include <cvsba/cvsba.h>

using namespace std;
using namespace boost;

const string imageDir = "./bin/data/dinoRing/";

//Given with Dataset.
 cv::Mat cameraIntrinsic = (cv::Mat_<float>(3,3) << 3310.400000f, 0.000000f, 316.730000f,
                                    0.000000f, 3325.500000f, 200.550000f,
                                    0.000000f, 0.000000f, 1.000000f);

//We'd normally assume world origin = intial camera position, but we're given it in the dino dataset, we use it here so we can check other camera poses.
glm::mat4 initialPose = glm::make_mat4(
    new float[16]{-0.08661715685291285200f, 0.97203145042392447000f, 0.21829465483805316000f, -0.97597093004059532000f, 
        -0.03881511324024737600f, -0.21441803766270939000f, -0.19994795321325870000f, -0.23162091782017200000f, 
        0.95203517502501223000f, -0.0526034704197f, 0.023290917003f, 0.659119498846f});

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

int main(int argc, const char* argv[]) {
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
    ImageData *previousImage = new ImageData(cv::String(filesystem::canonical(v[0]).string()), cameraIntrinsic,
                                colour, initialPose);
    v.erase(v.begin());
    
    vector<glm::vec3> cameraPosesToRender;
    glm::vec3 cameraPos = (vec4(1.0, 1.0, 1.0, 0.0) * previousImage->worldTransformation).xyz();
    cameraPosesToRender.push_back(cameraPos);
    
    //Create image pairs.
    for (vector<filesystem::path>::const_iterator itr = v.begin(); itr != v.end(); ++itr) {
        cv::String filePath = cv::String(filesystem::canonical(*itr).string()); //Get full file path, not relative.

        ImageData *currentImage = new ImageData(filePath, cameraIntrinsic, colour);
        imageSets.push_back(new ImageDataSet(previousImage, currentImage));

        cameraPos = (vec4(1.0, 1.0, 1.0, 0.0) * currentImage->worldTransformation).xyz();
        cameraPosesToRender.push_back(cameraPos);

        previousImage = currentImage;
    }

    cout << "Camera Positions: " << endl;
    for (vector<glm::vec3>::const_iterator itr = cameraPosesToRender.begin(); itr != cameraPosesToRender.end(); ++itr) {
        cout << glm::to_string(*itr) << endl;
    }


    cout << "Camera Poses: " << endl;
    cout << glm::to_string(imageSets[0]->image1->worldTransformation) << endl;
    for (vector<ImageDataSet*>::const_iterator itr = imageSets.begin(); itr != imageSets.end(); ++itr) {
        cout << glm::to_string((*itr)->image2->worldTransformation) << endl;
    }

    //  * @param  points N x 3 object points
    //  * @param  imagePoints M x N x 2 image points for each camera and each points. The outer  vector has M elements and each element has N elements of Point2d .
    //  * @param  visibility M x N x 1 visibility matrix, the element [i][j] = 1 when object point i is visible from camera j and 0 if not.
    //  * @param  cameraMatrix M x 3 x 3 camera matrix (intrinsic parameters) 3 x 3 camera matrix for each image
    //  * @param  distCoeffs M x   5  x1  distortion coefficient  for each image
    //  * @param R  M x 3 x 3 rotation matrix  for each image
    //  * @param T M x 3 x 1 translation matrix  for each image

    std::vector<cv::Point3f> points3D;    //3D Points
    std::vector<std::vector<cv::Point2f> >  imagePoints;    //List of a list of each images points
    std::vector<std::vector<int> > visibility;  //for each image, is each 3D 
    std::vector<cv::Mat> cameraMatrix;  //The intrinsic matrix for each camera.
    std::vector<cv::Mat> cameraRotations;
    std::vector<cv::Mat> cameraTranslations;
    std::vector<cv::Mat> distortionCoeffs;



    /*
        create new image_data_set, detecting points
        for each key feature in the new image
            add to imagePoints
            if it exists in the last image pair, it's part of a point track
                add to point track
            else
                triangulate point 
                add point
                 

        create new image_data_set, detecting points
        for each key feature in the new image
            add to imagePoints
            if it is part of a point track
                add to point track
            else
                triangulate point 
                add point

        triangulate points for new image_data_set
        for each newly triangulated point, check for point tracks
        if point track exists, add 
    */


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
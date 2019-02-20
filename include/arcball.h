//
//  Arcball.h
//  Arcball
//
//  Created by Saburo Okita on 12/03/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//  Modified by Tom Pritchard on 20/02/19

#ifndef __Arcball__Arcball__
#define __Arcball__Arcball__

#include <iostream>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/matrix_inverse.hpp>

class Arcball {
private:
    int windowWidth;
    int windowHeight;
    int leftMouseButtonDown;
    GLfloat rollSpeed;
    GLfloat angle ;
    glm::vec3 prevPos = glm::vec3(0.0f);
    glm::vec3 currPos = glm::vec3(0.0f);
    glm::vec3 camAxis;
    
    bool xAxis;
    bool yAxis;
    
public:
    Arcball( int window_width, int window_height, GLfloat roll_speed = 1.0f, bool x_axis = true, bool y_axis = true );
    glm::vec3 toScreenCoord( double x, double y );
    
    void mouseButtonCallback( GLFWwindow * window, int button, int action, int mods );
    void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    void cursorCallback( GLFWwindow *window, double x, double y );
    void update(float deltaTime);
    
    glm::mat4 createViewRotationMatrix();
    glm::mat4 createModelRotationMatrix( glm::mat4& view_matrix );
    
};

#endif /* defined(__Arcball__Arcball__) */
#ifndef VIEWSPACEMANIPULATOR_HPP
#define VIEWSPACEMANIPULATOR_HPP

    // #include "stdafx.h"
    #include <GL/glew.h>
    #include <GLFW/glfw3.h>
    #include <glm/glm.hpp>
    #include <glm/gtc/matrix_transform.hpp>

    using namespace glm;

    class viewspaceManipulator {
        public:
            viewspaceManipulator(GLFWwindow *window, vec3 initialCameraPos);
            mat4 getViewMatrix(); 
            mat4 getProjectionMatrix(); 
            void setup(GLFWwindow *window);  
            void update(GLFWwindow *window);    
            static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    };

#endif
#ifndef RENDERABLE_HPP
#define RENDERABLE_HPP

    // #include "stdafx.h"
    #include <vector>
    #include <GL/glew.h>
    #include <GLFW/glfw3.h>
    #define GLM_ENABLE_EXPERIMENTAL
    #include <glm/glm.hpp>
    #include <glm/gtc/matrix_transform.hpp>
    #include <glm/gtx/transform.hpp>

    using namespace glm;
    using namespace std;

    class Renderable {
        public:
            Renderable();
            Renderable(GLuint Shader, GLuint renderPrimative = GL_POINTS);
            Renderable(GLuint Shader, vector<glm::vec3> vert_data, GLuint renderPrimative = GL_POINTS);
            Renderable(GLuint Shader, vector<glm::vec3> vert_data, vector<glm::vec3> colour_data, GLuint renderPrimative = GL_POINTS);

            virtual ~Renderable() {}

            virtual GLuint getVAO();

            virtual void Draw(float deltaT, glm::mat4 projectionMatrix, glm::mat4 viewMatrix);

            vector<vec3> vertexes, colours;
	        GLuint pos_vbo, col_vbo, vao, shader, renderType;
            glm::mat4 modelMatrix = glm::mat4(1.0f);
            glm::mat4 scaleMatrix = glm::scale(glm::vec3(10.0, 10.0, 10.0));
            

            bool validVAO = false, isDead = false;
    };

#endif
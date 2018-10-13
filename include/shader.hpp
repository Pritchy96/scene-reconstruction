
#ifndef SHADER_HPP
#define SHADER_HPP

    #include <GL/glew.h>
    namespace Shader {
        GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path);
        GLuint LoadTransformShader(const char * path);
    }
#endif
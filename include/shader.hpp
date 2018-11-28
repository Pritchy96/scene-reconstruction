
#ifndef SHADER_HPP
#define SHADER_HPP

    #include <GL/glew.h>
    namespace Shader {
        GLuint LoadShaders(char * vertex_file_path, char * fragment_file_path);
        GLuint LoadTransformShader( char * path);
    }
#endif
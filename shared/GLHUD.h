#pragma once
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#endif

#ifdef WIN32
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#endif
#include <string>

namespace aphid {

class BaseCamera;
class GLHUD {
public:
    GLHUD();
    virtual ~GLHUD();
    void reset();
    void setCamera(BaseCamera * cam);
    void drawString(const std::string & str, const int & row) const;
private:
    static GLuint m_texture;
    BaseCamera * m_camera;
};

}

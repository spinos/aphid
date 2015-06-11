#ifndef ADENIUMWORLD_H
#define ADENIUMWORLD_H
#include <gl_heads.h>
class BvhTriangleSystem;
class BvhBuilder;
class TriangleSystem;
class AdeniumRender;
class BaseCamera;
class AdeniumWorld {
public:
    AdeniumWorld();
    virtual ~AdeniumWorld();
    
    void setBvhBuilder(BvhBuilder * builder);
    void addTriangleSystem(BvhTriangleSystem * tri); 
    void initOnDevice();
    void draw(BaseCamera * camera);
    void resizeRenderArea(int w, int h);
	void render(BaseCamera * camera);
private:
    void drawTriangle(TriangleSystem * tri);
private:
    BvhTriangleSystem * m_objects[32];
    AdeniumRender * m_image;
    unsigned m_numObjects;
	static GLuint m_texture;
};
#endif        //  #ifndef ADENIUMWORLD_H


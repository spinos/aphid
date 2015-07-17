#ifndef ADENIUMWORLD_H
#define ADENIUMWORLD_H
#include <gl_heads.h>
class BvhTriangleSystem;
class CUDABuffer;
class BvhBuilder;
class TriangleSystem;
class AdeniumRender;
class BaseCamera;
class WorldDbgDraw;
class ATriangleMesh;
class TriangleDifference;
class AdeniumWorld {
public:
    AdeniumWorld();
    virtual ~AdeniumWorld();
    
    void setBvhBuilder(BvhBuilder * builder);
    void addTriangleSystem(BvhTriangleSystem * tri); 
    void initOnDevice();
    void draw(BaseCamera * camera);
	void dbgDraw();
    void resizeRenderArea(int w, int h);
	void render(BaseCamera * camera);
	
	static WorldDbgDraw * DbgDrawer;
    
    void setRestMesh(ATriangleMesh * m);
    bool matchRestMesh(ATriangleMesh * m);
private:
    void drawTriangle(TriangleSystem * tri);
private:
    TriangleDifference * m_difference;
    BvhTriangleSystem * m_objects[32];
	CUDABuffer * m_objectInd[32];
	CUDABuffer * m_objectPos[32];
	CUDABuffer * m_objectVel[32];
	AdeniumRender * m_image;
	unsigned m_numObjects;
	static GLuint m_texture;
};
#endif        //  #ifndef ADENIUMWORLD_H


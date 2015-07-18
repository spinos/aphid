#ifndef ADENIUMWORLD_H
#define ADENIUMWORLD_H
#include <gl_heads.h>
#include <Vector3F.h>
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
    void setDifferenceObject(ATriangleMesh * m);
    ATriangleMesh * deformedMesh();
    void deform(bool toReset);
    bool isRayCast() const;
    void toggleRayCast();
    
    const Vector3F currentTranslation() const;
private:
    void drawTriangle(TriangleSystem * tri);
    void drawOverallTranslation();
private:
    TriangleDifference * m_difference;
    BvhTriangleSystem * m_objects[32];
	CUDABuffer * m_objectInd[32];
	CUDABuffer * m_objectPos[32];
	CUDABuffer * m_objectVel[32];
	AdeniumRender * m_image;
    ATriangleMesh * m_deformedMesh;
	unsigned m_numObjects;
	static GLuint m_texture;
    bool m_enableRayCast;
};
#endif        //  #ifndef ADENIUMWORLD_H


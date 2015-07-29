#ifndef ADENIUMWORLD_H
#define ADENIUMWORLD_H
#include <gl_heads.h>
#include <Matrix44F.h>

class BvhTriangleSystem;
class CUDABuffer;
class BvhBuilder;
class TriangleSystem;
class AdeniumRender;
class BaseCamera;
class WorldDbgDraw;
class ATriangleMesh;
class TriangleDifference;
class ATetrahedronMesh;
class TriangleAnchorDeformer;
class OVelocityFile;
class AFrameRange;
class AdeniumWorld {
public:
    AdeniumWorld();
    virtual ~AdeniumWorld();
    
    void setBvhBuilder(BvhBuilder * builder);
    void addTriangleSystem(BvhTriangleSystem * tri); 
    void addTetrahedronMesh(ATetrahedronMesh * tetra);
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
    void processFrame(bool toReset);
    bool isRayCast() const;
    void toggleRayCast();
    
    const Vector3F currentTranslation() const;
    void beginRecordVelocity(AFrameRange * fr);
    bool doneVelocityCaching() const;
private:
    void drawTriangle(TriangleSystem * tri);
    void drawTetrahedron();
    void drawAnchors();
    void drawOverallTranslation();
    bool saveVelocityFramerange(AFrameRange * fr);
    void saveVelocity(bool toReset);
	void recordT();
    void recordP();
    const unsigned totalNumPoints() const;
    
private:
    Matrix44F m_restSpaceInv;
    TriangleDifference * m_difference;
    BvhTriangleSystem * m_objects[32];
	CUDABuffer * m_objectInd[32];
	CUDABuffer * m_objectPos[32];
	CUDABuffer * m_objectVel[32];
    CUDABuffer * m_objectAnchoredVel[32];
	AdeniumRender * m_image;
    ATriangleMesh * m_deformedMesh;
    ATetrahedronMesh * m_tetraMesh;
    TriangleAnchorDeformer * m_tetraDeformer;
    OVelocityFile * m_velocityOut;
	unsigned m_numObjects;
	static GLuint m_texture;
    bool m_enableRayCast;
};
#endif        //  #ifndef ADENIUMWORLD_H


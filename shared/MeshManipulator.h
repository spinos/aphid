#pragma once

#include <AllMath.h>
class Ray;
class BaseMesh;
class IntersectionContext;
class MeshTopology;
class MeshManipulator {
public:
    MeshManipulator();
    virtual ~MeshManipulator();
    void attachTo(BaseMesh * mesh);
    
    void start(const Ray * r);
	void perform(const Ray * r);
	void stop();
	
	void setToMove();
	void setToSmooth();
private:
	void moveVertex(const Ray * r);
	void smoothSurface(const Ray * r);

private:
    BaseMesh * m_mesh;
	MeshTopology * m_topo;
    IntersectionContext * m_intersect;
    char m_started;
	int m_mode;
};

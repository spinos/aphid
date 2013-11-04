#pragma once

#include <AllMath.h>
class Ray;
class BaseMesh;
class IntersectionContext;
class MeshManipulator {
public:
    MeshManipulator();
    virtual ~MeshManipulator();
    void attachTo(BaseMesh * mesh);
    
    void start(const Ray * r);
	void perform(const Ray * r);
	void stop();
private:
    BaseMesh * m_mesh;
    IntersectionContext * m_intersect;
    char m_started;
};

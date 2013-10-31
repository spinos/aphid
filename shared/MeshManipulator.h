#pragma once

#include <AllMath.h>
class Ray;
class BaseMesh;
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
    char m_started;
};

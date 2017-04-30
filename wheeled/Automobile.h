#pragma once
#include "WheeledVehicle.h"

namespace caterpillar {
class Mesh;
class Automobile : public WheeledVehicle {
public:
    Automobile();
    virtual ~Automobile();
    
    void render();
private:
    void fillMesh(Mesh * m, 
        const int & nv, const int & ntv, 
        const int * indices,
        const float * pos, const float * nor) const;
    void drawMesh(const Matrix44F & mat, Mesh * m);
private:
    Mesh * m_chassisMesh;
    Mesh * m_wheelMesh[2];
};
}


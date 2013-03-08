#ifndef MESHLAPLACIAN_H
#define MESHLAPLACIAN_H

#include "TriangleMesh.h"

class MeshLaplacian : public TriangleMesh {
public:
    MeshLaplacian();
    MeshLaplacian(const char * filename);
    virtual ~MeshLaplacian();
    
private:
    
};
#endif        //  #ifndef MESHLAPLACIAN_H


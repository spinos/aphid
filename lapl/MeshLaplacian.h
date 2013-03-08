#ifndef MESHLAPLACIAN_H
#define MESHLAPLACIAN_H

#include "TriangleMesh.h"
#include <vector>
class VertexAdjacency;
class MeshLaplacian : public TriangleMesh {
public:
    MeshLaplacian();
    MeshLaplacian(const char * filename);
    virtual ~MeshLaplacian();
    
private:
    VertexAdjacency * m_adjacency;
};
#endif        //  #ifndef MESHLAPLACIAN_H


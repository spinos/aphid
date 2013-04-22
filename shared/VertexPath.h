#pragma once

#include <VertexAdjacency.h>

class VertexPath {
public:
    VertexPath();
    virtual ~VertexPath();
    
    void setTopology(VertexAdjacency * topo);
    void create(unsigned startVert, unsigned endVert);
private:
    char recursiveFindClosestNeighbor(unsigned vert, unsigned endVert, const Vector3F & endPoint);
    std::vector<unsigned> m_vertices;
    VertexAdjacency * m_topology;
};


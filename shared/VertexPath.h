#pragma once

#include <VertexAdjacency.h>

class VertexPath {
public:
    VertexPath();
    virtual ~VertexPath();
    
    void setTopology(VertexAdjacency * topo);
    void create(unsigned startVert, unsigned endVert);
	unsigned grow(unsigned startVert, unsigned endVert);
	
	unsigned numVertices() const;
	unsigned vertex(unsigned idx) const;
private:
    char recursiveFindClosestNeighbor(unsigned vert, unsigned endVert, const Vector3F & endPoint);
    std::vector<unsigned> m_vertices;
    VertexAdjacency * m_topology;
};


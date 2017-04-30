#pragma once
#include <vector>
#include <math/Vector3F.h>

namespace aphid {

class MeshTopology;

class VertexPath {
public:
    VertexPath();
    virtual ~VertexPath();
    
    void setTopology(MeshTopology * topo);
    void create(unsigned startVert, unsigned endVert);
	bool grow(unsigned startVert, unsigned endVert, unsigned & dst);
	bool growOnBoundary(unsigned startVert, unsigned endVert, unsigned & dst);
	
	unsigned numVertices() const;
	unsigned vertex(unsigned idx) const;
private:
    char recursiveFindClosestNeighbor(unsigned vert, unsigned endVert, const Vector3F & endPoint);
    std::vector<unsigned> m_vertices;
    MeshTopology * m_topology;
};

}


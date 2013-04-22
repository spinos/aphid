#include <VertexPath.h>

VertexPath::VertexPath() {}
VertexPath::~VertexPath() {}

void VertexPath::setTopology(VertexAdjacency * topo)
{
    m_topology = topo;
}

void VertexPath::create(unsigned startVert, unsigned endVert)
{
    m_vertices.clear();
    VertexAdjacency adj = m_topology[startVert];
    
    Vector3F endPoint = m_topology[endVert]->m_v;
    recursiveFindClosestNeighbor(startVert, endVert, endPoint);
}

char VertexPath::recursiveFindClosestNeighbor(unsigned vert, unsigned endVert, const Vector3F & endPoint)
{
    unsigned closestNei;
    float minDist = 10e8;
    for(VertexNeighbor * nei = adj.firstNeighbor(); !adj.isLastNeighbor(); nei = nextNeighbor()) {
        if(nei->v->getIndex() == endVert) {
            return 1;
        }
        
        Vector3F p = *(nei->v->m_v);
        float dist = (p - endPoint).length();
        if(dist < minDist) {
            minDist = dist;
            closestNei = nei->v->getIndex();
        }
    }
    
    return recursiveFindClosestNeighbor(closestNei, endVert, endPoint));
}

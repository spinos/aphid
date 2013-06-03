#include <VertexPath.h>
#include <MeshTopology.h>
#include <VertexAdjacency.h>

VertexPath::VertexPath() {}
VertexPath::~VertexPath() {}

void VertexPath::setTopology(MeshTopology * topo)
{
    m_topology = topo;
}

void VertexPath::create(unsigned startVert, unsigned endVert)
{
    m_vertices.clear();
    
    Vector3F endPoint = *(m_topology->getAdjacency(endVert).m_v);
    recursiveFindClosestNeighbor(startVert, endVert, endPoint);
}

char VertexPath::recursiveFindClosestNeighbor(unsigned vert, unsigned endVert, const Vector3F & endPoint)
{
	VertexAdjacency adj = m_topology->getAdjacency(vert);
	Vector3F vertP = *(adj.m_v);
    unsigned closestNei;
    float minDist = 10e8;
    for(VertexAdjacency::VertexNeighbor * nei = adj.firstNeighbor(); !adj.isLastNeighbor(); nei = adj.nextNeighbor()) {
        if(nei->v->getIndex() == endVert) {
			m_vertices.push_back(endVert);
            return 1;
        }
        
        Vector3F p = *(nei->v->m_v);
        float dist = (p - endPoint).length() + (vertP - p).length();
        if(dist < minDist) {
            minDist = dist;
            closestNei = nei->v->getIndex();
        }
    }
    m_vertices.push_back(closestNei);
	if(m_vertices.size() > 64) return 1;
    return recursiveFindClosestNeighbor(closestNei, endVert, endPoint);
}

bool VertexPath::grow(unsigned startVert, unsigned endVert, unsigned & dst)
{
	VertexAdjacency &adj = m_topology->getAdjacency(endVert);
	if(!adj.isConnectedTo(startVert)) return false;
	if(adj.isOpen()) return growOnBoundary(startVert, endVert, dst);
	
	dst = adj.nextRealEdgeNeighbor(startVert);
	dst = adj.nextRealEdgeNeighbor(dst);
	return true;
}

bool VertexPath::growOnBoundary(unsigned startVert, unsigned endVert, unsigned & dst)
{
	VertexAdjacency &adj = m_topology->getAdjacency(startVert);
	if(!adj.isOpen()) return false;
	adj = m_topology->getAdjacency(endVert);
	if(!adj.isOpen()) return false;
	
	dst = adj.nextBoundaryNeighbor(startVert);
	return true;
}

unsigned VertexPath::numVertices() const
{
	return (unsigned)m_vertices.size();
}

unsigned VertexPath::vertex(unsigned idx) const
{
	return m_vertices[idx];
}


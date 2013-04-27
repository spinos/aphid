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
    
    Vector3F endPoint = *(m_topology[endVert].m_v);
    recursiveFindClosestNeighbor(startVert, endVert, endPoint);
}

char VertexPath::recursiveFindClosestNeighbor(unsigned vert, unsigned endVert, const Vector3F & endPoint)
{
	VertexAdjacency adj = m_topology[vert];
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
	VertexAdjacency adj = m_topology[endVert];
	if(adj.isOpen()) return false;
	if(!adj.isConnectedTo(startVert)) return false;
	
	dst = adj.nextRealEdgeNeighbor(startVert);
	dst = adj.nextRealEdgeNeighbor(dst);
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


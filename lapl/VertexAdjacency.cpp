/*
 *  VertexAdjacency.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/9/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "VertexAdjacency.h"
#include "Edge.h"
#include "Facet.h"
#include <cmath>
VertexAdjacency::VertexAdjacency() {}
VertexAdjacency::~VertexAdjacency() 
{
	m_edges.clear();
}

void VertexAdjacency::addEdge(Edge * e)
{
	m_edges.push_back(e);
}

char VertexAdjacency::findOneRingNeighbors()
{
	int nneib = 0;
	Edge outgoing;
	firstOutgoingEdge(outgoing);
	m_neighbors.push_back(outgoing.v1());
	m_idxInOrder[outgoing.v1()->getIndex()] = nneib;
	nneib++;
	Edge incoming;
	if(!findIncomming(outgoing, incoming)) {
		printf("cannot find incoming edge ");
		return 0;
	}
	//printf("e %i-%i ", incoming.v0()->getIndex(), incoming.v1()->getIndex());
	
	if(!findOppositeEdge(incoming, outgoing)) {
		printf("cannot find outgoing edge ");
		return 0;
	}
	for(int i = 1; i < m_edges.size() / 2; i++) {
		m_neighbors.push_back(outgoing.v1());
		m_idxInOrder[outgoing.v1()->getIndex()] = nneib;
		nneib++;
		findIncomming(outgoing, incoming);
		findOppositeEdge(incoming, outgoing);
	}
	return 1;
}

void VertexAdjacency::computeWeights()
{
	Vector3F vij, vij0, vij1;
	float dist, theta0, theta1, wij;

	const unsigned numNeighbors = m_neighbors.size();
	for(unsigned i = 0; i < numNeighbors; i++) {
		getVijs(i, vij, vij0, vij1);
		
		dist = vij.length();
		vij.normalize();
		vij0.normalize();
		vij1.normalize();
		
		theta0 = acos(vij.dot(vij0));
		theta1 = acos(vij.dot(vij1));
		
		wij = (tan(theta0 * 0.5f) + tan(theta1 * 0.5f))/dist;
		m_weights.push_back(wij);
	}
	
	float sum = 0;
	for(unsigned i = 0; i < numNeighbors; i++) {
		sum += m_weights[i];
	}
	
	for(unsigned i = 0; i < numNeighbors; i++) {
		m_weights[i] /= sum;
	}
	
	m_mvcoord = Vector3F(0.f, 0.f, 0.f);
	
	for(unsigned i = 0; i < numNeighbors; i++) {
		m_mvcoord += *m_neighbors[i] * m_weights[i];
	}
	
	m_mvcoord -= *this;
}

void VertexAdjacency::computeNormal()
{
    Vector3F vij, vij0, vij1, faceN;
    float faceArea;
    m_normal = Vector3F(0.f, 0.f, 0.f);
    const unsigned numNeighbors = m_neighbors.size();
	for(unsigned i = 0; i < numNeighbors; i++) {
		getVijs(i, vij, vij0, vij1);
		
		vij.normalize();
		vij1.normalize();
		
		faceN = vij.cross(vij1);
		faceN.normalize();
		
		faceArea = Facet::cumputeArea((Vector3F *)this, &vij, &vij1);
		
		m_normal += faceN * faceArea;
	}
	m_normal.normalize();
}

char VertexAdjacency::findOppositeEdge(Edge & e, Edge &dest) const
{
	
	std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		//printf("e %i-%i ", (*it)->v0()->getIndex(), (*it)->v1()->getIndex());
		if((*it)->isOppositeOf(&e)) {
			dest = *(*it);
			return 1;
		}
	}
	return 0;
}

char VertexAdjacency::firstOutgoingEdge(Edge & e)
{
	std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		if((*it)->v0()->getIndex() == getIndex()) {
			e = *(*it);
			return 1;
		}
	}
	return 0;
}

char VertexAdjacency::findIncomming(Edge & eout, Edge & ein)
{
	const int faceId = ((Facet *)eout.getFace())->getIndex();
	std::vector<Edge *>::iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		//printf("ee %i-%i ", (*it)->v0()->getIndex(), (*it)->v1()->getIndex());
	
		if((*it)->v1()->getIndex() == getIndex() && ((Facet *)(*it)->getFace())->getIndex() == faceId) {
			ein = *(*it);
			
			return 1;
		}
	}
	return 0;
}

std::map<int,int> VertexAdjacency::getNeighborOrder() const
{
	return m_idxInOrder;
}

void VertexAdjacency::getNeighbor(const int & idx, int & vertexIdx, float & weight) const
{
	vertexIdx = m_neighbors[idx]->getIndex();
	weight = m_weights[idx];
}

float VertexAdjacency::getDeltaCoordX() const
{
	return m_mvcoord.x;
}
	
float VertexAdjacency::getDeltaCoordY() const
{
	return m_mvcoord.y;
}
	
float VertexAdjacency::getDeltaCoordZ() const
{
	return m_mvcoord.z;
}

void VertexAdjacency::getVijs(const int & idx, Vector3F &vij, Vector3F &vij0, Vector3F &vij1) const
{
    const int numNeighbors = (int)m_neighbors.size();
    vij = *m_neighbors[idx] - *this;
    if(idx == 0)
        vij0 = *m_neighbors[numNeighbors - 1] - *this;
    else
        vij0 = *m_neighbors[idx - 1] - *this;
    if(idx == numNeighbors - 1)
        vij1 = *m_neighbors[0] - *this;
    else
        vij1 = *m_neighbors[idx + 1] - *this;
}

void VertexAdjacency::verbose() const
{
	printf("\nv %i\n adjacent edge count: %i\n", getIndex(), (int)m_edges.size());
	/*std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		printf(" %d - %d", (*it)->v0()->getIndex(), (*it)->v1()->getIndex());
	}*/
	std::vector<Vertex *>::const_iterator it;
	for(it = m_neighbors.begin(); it < m_neighbors.end(); it++) {
		printf(" %i ", (*it)->getIndex());
	}
	printf("\n");
	std::vector<float >::const_iterator itw;
	for(itw = m_weights.begin(); itw < m_weights.end(); itw++) {
		printf(" %f ", (*itw));
	}
	printf("\n delta-coordinate %f %f %f \n", m_mvcoord.x, m_mvcoord.y, m_mvcoord.z);
}
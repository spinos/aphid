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
#include <math/Matrix44F.h>

#include <cmath>

namespace aphid {

VertexAdjacency::VertexAdjacency() {}
VertexAdjacency::~VertexAdjacency() 
{
	m_edges.clear();
}

void VertexAdjacency::addEdge(Edge * e)
{
	m_edges.push_back(e);
}

char VertexAdjacency::isOpen() const
{
    Edge dummy;
    return firstOutgoingEdgeOnBoundary(dummy);
}

void VertexAdjacency::findNeighbors()
{
    if(!isOpen())
        findOneRingNeighbors();
    else
		findOpenNeighbors();
}

void VertexAdjacency::findOpenNeighbors()
{
	Edge outgoing;
	firstOutgoingEdgeOnBoundary(outgoing);
	addNeighbor(&outgoing);
	
	Edge incoming;
	findIncomming(outgoing, incoming);
	
	while(findOppositeEdge(incoming, outgoing)) {
		addNeighbor(&outgoing);
		findIncomming(outgoing, incoming);
	}
	
	addNeighbor(&incoming, 0);
}

char VertexAdjacency::findOneRingNeighbors()
{
	Edge outgoing;
	firstOutgoingEdge(outgoing);
	
	addNeighbor(&outgoing);

	Edge incoming;
	if(!findIncomming(outgoing, incoming)) {
		//printf("cannot find incoming edge ");
		return 0;
	}
	//printf("e %i-%i ", incoming.v0()->getIndex(), incoming.v1()->getIndex());
	
	if(!findOppositeEdge(incoming, outgoing)) {
		//printf("cannot find outgoing edge ");
		return 0;
	}
	
	for(int i = 1; i < (int)m_edges.size() / 2; i++) {
		addNeighbor(&outgoing);
		findIncomming(outgoing, incoming);
		findOppositeEdge(incoming, outgoing);
	}
	return 1;
}

void VertexAdjacency::computeWeights()
{
	const unsigned numNeighbors = getNumNeighbors();
	if(isOpen()) {
		for(unsigned i = 0; i < numNeighbors; i++) {
			m_neighbors[i]->weight = 1.f / (float)numNeighbors;
		}
		return;
	}
	
	Vector3F vij, vij0, vij1;
	float dist, theta0, theta1, wij;

	for(unsigned i = 0; i < numNeighbors; i++) {
		getVijs(i, vij, vij0, vij1);
		
		dist = vij.length();
		vij.normalize();
		vij0.normalize();
		vij1.normalize();
		
		theta0 = acos(vij.dot(vij0));
		theta1 = acos(vij.dot(vij1));
		
		wij = (tan(theta0 * 0.5) + tan(theta1 * 0.5))/dist;
		m_neighbors[i]->weight = wij;
	}
	
	float sum = 0;
	for(unsigned i = 0; i < numNeighbors; i++) {
		sum += m_neighbors[i]->weight;
	}
	
	for(unsigned i = 0; i < numNeighbors; i++) {
		m_neighbors[i]->weight /= sum;
	}
}

void VertexAdjacency::computeDifferentialCoordinate()
{
	m_differential.setZero();
	const unsigned numNeighbors = m_neighbors.size();
	for(unsigned i = 0; i < numNeighbors; i++) {
		Vector3F vij = *m_v - *(m_neighbors[i]->v->m_v);
		m_differential += vij * m_neighbors[i]->weight;
	}
}

void VertexAdjacency::computeTangentFrame()
{
    const unsigned numNeighbors = m_neighbors.size();
	
    m_normal = Vector3F(0.f, 0.f, 0.f);
    for(unsigned i = 0; i < numNeighbors; i++) {
		Facet *f = m_neighbors[i]->f;
		m_normal += f->getNormal() * f->getArea();
	}
	m_normal.normalize();
	
	const Vector3F x1 = *(m_neighbors[0]->v->m_v) - *m_v;
	
	Vector3F binormal = x1.cross(m_normal);
	binormal.normalize();
	
	Vector3F tangent = m_normal.cross(binormal);
	tangent.normalize();
	
	m_tangentFrame.fill(tangent, binormal, m_normal);
}

Vector3F VertexAdjacency::computeNormal()
{
	Vector3F s;
	s.setZero();
	const unsigned numNeighbors = m_neighbors.size();
	for(unsigned i = 0; i < numNeighbors; i++) {
		Facet *f = m_neighbors[i]->f;
		s += f->getNormal() * f->getArea();
	}
	s.normalize();
	return s;
}

char VertexAdjacency::findOppositeEdge(int i, int j, Edge & dest) const
{
	std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		if((*it)->isOppositeOf(i, j)) {
			dest = *(*it);
			return 1;
		}
	}
	return 0;
}

char VertexAdjacency::findEdge(int i, int j, Edge & dest) const
{
	std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		if((*it)->isOppositeOf(i, j) || (*it)->isOppositeOf(j, i)) {
			dest = *(*it);
			return 1;
		}
	}
	return 0;
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

char VertexAdjacency::firstOutgoingEdge(Edge & e) const
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

char VertexAdjacency::firstOutgoingEdgeOnBoundary(Edge & e) const
{
    Edge dummy;
    std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		if((*it)->v0()->getIndex() == getIndex()) {
			e = *(*it);
			if(!findOppositeEdge(e, dummy))
			    return 1;
		}
	}
	return 0;
}

char VertexAdjacency::findIncomming(Edge & eout, Edge & ein) const
{
	const int faceId = ((Facet *)eout.getFace())->getIndex();
	std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		//printf("ee %i-%i ", (*it)->v0()->getIndex(), (*it)->v1()->getIndex());
	
		if((*it)->v1()->getIndex() == getIndex() && ((Facet *)(*it)->getFace())->getIndex() == faceId) {
			ein = *(*it);
			
			return 1;
		}
	}
	return 0;
}

unsigned VertexAdjacency::getNumNeighbors() const
{
    return (unsigned)m_neighbors.size();
}

void VertexAdjacency::addNeighbor(Edge *e, char isOutgoing)
{
	VertexNeighbor *aneighbor = new VertexNeighbor();
	if(isOutgoing)
		aneighbor->v = e->v1();
	else
		aneighbor->v = e->v0();
		
	aneighbor->f = (Facet *)e->getFace();
	aneighbor->weight = 1.f;
	
	m_neighbors.push_back(aneighbor);
	if(isOutgoing)
		m_idxInOrder[e->v1()->getIndex()] = m_idxInOrder.size();
	else
		m_idxInOrder[e->v0()->getIndex()] = m_idxInOrder.size();
}

void VertexAdjacency::getVijs(int idx, Vector3F &vij, Vector3F &vij0, Vector3F &vij1) const
{
    const int numNeighbors = (int)m_neighbors.size();
	Vertex * v = m_neighbors[idx]->v;
    vij = *(v->m_v) - *m_v;

	if(idx == 0) {
		vij0 = *(m_neighbors[numNeighbors - 1]->v->m_v) - *m_v;
		vij1 = *(m_neighbors[idx + 1]->v->m_v) - *m_v;
	}
    else if(idx == numNeighbors - 1) {
		vij0 = *(m_neighbors[idx - 1]->v->m_v) - *m_v;
		vij1 = *(m_neighbors[0]->v->m_v) - *m_v;
	}    
    else {
		vij0 = *(m_neighbors[idx - 1]->v->m_v) - *m_v;
        vij1 = *(m_neighbors[idx + 1]->v->m_v) - *m_v;
	}
}

Matrix33F VertexAdjacency::getTangentFrame() const
{
    return m_tangentFrame;
}

Vector3F VertexAdjacency::getDifferentialCoordinate() const
{
	return m_differential;
}

VertexAdjacency::VertexNeighbor * VertexAdjacency::firstNeighbor()
{
    m_neighborIt = m_neighbors.begin();
    return *m_neighborIt;
}

VertexAdjacency::VertexNeighbor * VertexAdjacency::nextNeighbor()
{
    m_neighborIt++;
    if(isLastNeighbor()) return 0;
    return *m_neighborIt;
}

char VertexAdjacency::isLastNeighbor()
{
    return m_neighborIt == m_neighbors.end();
}

VertexAdjacency::VertexNeighbor * VertexAdjacency::firstNeighborOrderedByVertexIdx()
{
    m_orderedNeighborIt = m_idxInOrder.begin();
    return m_neighbors[m_orderedNeighborIt->second];
}

VertexAdjacency::VertexNeighbor * VertexAdjacency::nextNeighborOrderedByVertexIdx()
{
    m_orderedNeighborIt++;
    if(isLastNeighborOrderedByVertexIdx()) return 0;
    return m_neighbors[m_orderedNeighborIt->second];
}

char VertexAdjacency::isLastNeighborOrderedByVertexIdx()
{
    return m_orderedNeighborIt == m_idxInOrder.end();
}

bool VertexAdjacency::isConnectedTo(unsigned idx)
{
	for(m_neighborIt = m_neighbors.begin(); m_neighborIt != m_neighbors.end(); ++m_neighborIt) {
		if((*m_neighborIt)->v->getIndex() == idx) 
			return true;
	}
	return false;
}

Edge * VertexAdjacency::outgoingEdgeToVertex(unsigned idx, char & res)
{
	res = 0;
	std::vector<Edge *>::iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		if((*it)->isReal() && (*it)->v1()->getIndex() == idx) {
			res = 1;
			return *it;
		}
	}
	return 0;
}

Edge * VertexAdjacency::connectedToVertexBy(unsigned idx, char & res)
{
	res = 0;
	std::vector<Edge *>::iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		if((*it)->isReal() && (*it)->connectedToVertex(idx)) {
			res = 1;
			return *it;
		}
	}
	return 0;
}

unsigned VertexAdjacency::nextRealEdgeNeighbor(unsigned idx)
{
	for(m_neighborIt = m_neighbors.begin(); m_neighborIt != m_neighbors.end(); ++m_neighborIt) {
		if((*m_neighborIt)->v->getIndex() == idx) {
			m_neighborIt++;
			if(m_neighborIt == m_neighbors.end()) m_neighborIt = m_neighbors.begin();
			break;
		}
	}
	
	Edge dummy;
	for(unsigned i = 0; i < getNumNeighbors()-1; i++) {
		findEdge(this->getIndex(), (*m_neighborIt)->v->getIndex(), dummy);
		if(dummy.isReal()) return (*m_neighborIt)->v->getIndex();
		m_neighborIt++;
		if(m_neighborIt == m_neighbors.end()) m_neighborIt = m_neighbors.begin();
	}
	
	return (*m_neighborIt)->v->getIndex();
}

unsigned VertexAdjacency::nextBoundaryNeighbor(unsigned idx)
{
	for(m_neighborIt = m_neighbors.begin(); m_neighborIt != m_neighbors.end(); ++m_neighborIt) {
		if((*m_neighborIt)->v->getIndex() == idx) {
			m_neighborIt++;
			if(m_neighborIt == m_neighbors.end()) m_neighborIt = m_neighbors.begin();
			break;
		}
	}
	
	Edge dummy, nouse;
	for(unsigned i = 0; i < getNumNeighbors()-1; i++) {
		findEdge(this->getIndex(), (*m_neighborIt)->v->getIndex(), dummy);
		if(!findOppositeEdge(dummy, nouse)) return (*m_neighborIt)->v->getIndex();
		m_neighborIt++;
		if(m_neighborIt == m_neighbors.end()) m_neighborIt = m_neighbors.begin();
	}
	
	return (*m_neighborIt)->v->getIndex();
}

void VertexAdjacency::connectEdges()
{
	std::vector<Edge *>::iterator eit;
	std::vector<Edge *>::iterator eit1;
	for(eit = m_edges.begin(); eit < m_edges.end(); eit++) {
		for(eit1 = m_edges.begin(); eit1 < m_edges.end(); eit1++) {
			if((*eit1)->isOppositeOf(*eit)) {
				(*eit)->setTwin(*eit1);
			}
		}
	}
}

template<typename T>
inline bool IsElementIn(T const& a, const std::vector<T>& array)
{
	typename std::vector<T>::const_iterator it;
	for(it = array.begin(); it != array.end(); ++it) {
		if(a == *it) return true;
	}
	return false;
}

void VertexAdjacency::getConnectedFacets(std::vector<unsigned> & dst) const
{
	const unsigned numNeighbors = m_neighbors.size();
	for(unsigned i = 0; i < numNeighbors; i++) {
		Facet *f = m_neighbors[i]->f;
		const unsigned triIdx = f->getIndex();
		if(!IsElementIn(triIdx, dst)) dst.push_back(triIdx);
	}
}

void VertexAdjacency::getConnectedPolygons(std::vector<unsigned> & dst) const
{
	const unsigned numNeighbors = m_neighbors.size();
	for(unsigned i = 0; i < numNeighbors; i++) {
		Facet *f = m_neighbors[i]->f;
		unsigned polyIdx = f->getPolygonIndex();
		if(!IsElementIn(polyIdx, dst)) dst.push_back(polyIdx);
	}
}

Vector3F VertexAdjacency::center()
{
	char onBoundary = isOpen();
	int numNeiOnBoundary = 0;
	Vector3F c;
	VertexAdjacency::VertexNeighbor *neighbor;
    for(neighbor = firstNeighbor(); !isLastNeighbor(); neighbor = nextNeighbor()) {
        if(onBoundary) {
			if(isNeighborOnBoundary(*neighbor)) {
				c += *(neighbor->v->m_v);
				numNeiOnBoundary++;
			}
		}
		else
			c += *(neighbor->v->m_v) * neighbor->weight;
    }
	
	if(numNeiOnBoundary > 0) c /= (float)numNeiOnBoundary;
	return c;
}

char VertexAdjacency::isNeighborOnBoundary(VertexNeighbor & nei)
{
	Edge dummy, nouse;
	findEdge(this->getIndex(), nei.v->getIndex(), dummy);
	if(!findOppositeEdge(dummy, nouse)) return 1;
	return 0;
}

unsigned VertexAdjacency::numRealEdgeNeighbors() const
{
	unsigned r = 0;
	std::vector<VertexNeighbor *>::const_iterator it = m_neighbors.begin();
	Edge dummy;
	for(; it != m_neighbors.end(); ++it) {
		findEdge(this->getIndex(), (*it)->v->getIndex(), dummy);
		if(dummy.isReal()) r++;
	}
	return r;
}

void VertexAdjacency::verbose() const
{
	//printf("\nv %i\n adjacent edge count: %i\n", getIndex(), (int)m_edges.size());
	std::vector<Edge *>::const_iterator eit;
	for(eit = m_edges.begin(); eit < m_edges.end(); eit++) {
		//printf(" %d - %d f %i", (*eit)->v0()->getIndex(), (*eit)->v1()->getIndex(), ((Facet *)(*eit)->getFace())->getPolygonIndex());
	}
	std::vector<VertexNeighbor *>::const_iterator it;
	for(it = m_neighbors.begin(); it < m_neighbors.end(); it++) {
		//printf(" %i ", (*it)->v->getIndex());
	}
}

}

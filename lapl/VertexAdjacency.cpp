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
VertexAdjacency::VertexAdjacency() {}
VertexAdjacency::~VertexAdjacency() 
{
	m_edges.clear();
}

void VertexAdjacency::addEdge(Edge * e)
{
	m_edges.push_back(e);
}

char VertexAdjacency::checkOneRing() const
{
	Edge e;
	std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		Edge * exist = *it;
		if(!findOppositeEdge(*exist, e))
			return 0;
	}
	return 1;
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

char VertexAdjacency::findOneRingNeighbors()
{
	Edge outgoing;
	firstOutgoingEdge(outgoing);
	m_neighbors.push_back(outgoing.v1());
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
		findIncomming(outgoing, incoming);
		findOppositeEdge(incoming, outgoing);
	}
	return 1;
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

void VertexAdjacency::verbose() const
{
	printf(" adjacent edge count: %d", m_edges.size());
	/*std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		printf(" %d - %d", (*it)->v0()->getIndex(), (*it)->v1()->getIndex());
	}*/
	std::vector<Vertex *>::const_iterator it;
	for(it = m_neighbors.begin(); it < m_neighbors.end(); it++) {
		printf(" %i-", (*it)->getIndex());
	}
}
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
VertexAdjacency::VertexAdjacency() {}
VertexAdjacency::~VertexAdjacency() {}

void VertexAdjacency::addEdge(Edge * e, int idx)
{
	/*std::vector<Edge *>::iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		Edge * exist = *it;
		if(exist->matches(e))
			return;
	}*/
	if(e->v0()->getIndex() == idx)
	m_edges.push_back(e);
}

void VertexAdjacency::verbose() const
{
	printf(" adjacent edge count: %d", m_edges.size());
	std::vector<Edge *>::const_iterator it;
	for(it = m_edges.begin(); it < m_edges.end(); it++) {
		printf(" %d - %d", (*it)->v0()->getIndex(), (*it)->v1()->getIndex());
	}
}
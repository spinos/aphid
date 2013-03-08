/*
 *  VertexAdjacency.h
 *  lapl
 *
 *  Created by jian zhang on 3/9/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <vector>

class Edge;
class VertexAdjacency {
public:
	VertexAdjacency();
	virtual ~VertexAdjacency();
	
	void addEdge(Edge * e, int idx);
	
	void verbose() const;
private:
	std::vector<Edge *> m_edges;
};

/*
 *  ConflictGraph.h
 *  convexHull
 *
 *  Created by jian zhang on 9/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <GraphArch.h>
#include <vector>
class ConflictGraph {
public:
	ConflictGraph(char faceOriented);
	~ConflictGraph();
	void clear();
	void add(GraphArch * arch);
	void getFaces(std::vector<Facet *>&faces) const;
	void getVertices(std::vector<Vertex *>&vertices) const;
	void getVertices(GeoElement * dest) const;
	void removeFace(Facet *f);
private:
	GraphArch *m_head;
	char m_faceOriented;
};
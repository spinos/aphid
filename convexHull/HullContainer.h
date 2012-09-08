/*
 *  HullContainer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "shapeDrawer.h"
#include <Vertex.h>
#include <Facet.h>
#include <vector>

class HullContainer {
public:
	HullContainer();
	virtual ~HullContainer();
	
	int getNumVertex() const;
	int getNumFace() const;
	
	void addVertex(Vertex *p);
	Vertex getVertex(int idx) const;
	Vertex *vertex(int idx);
	
	void addFacet(Facet *f);
	Facet getFacet(int idx) const;
	void removeFaces();
	
	void initHull();
	void killHull();
	void renderWorld();

	void beginHull();
	char searchVisibleFaces(Vertex *v);
	char searchHorizons();
	char spawn(Vertex *v);
	char finishStep(Vertex *v);
	void addConflict(Facet *f, Vertex *v);
	void addConflict(Facet *f, Facet *a, Facet *b);
	void removeConflict(Facet *f);
	
protected:
	ShapeDrawer* fDrawer;
	std::vector<Vertex *>m_vertices;
	std::vector<Facet *>m_faces;
	std::vector<Facet *>visibleFaces;
	Edge * m_horizon;
	int m_currentVertexId;
	int m_numHorizon;
};
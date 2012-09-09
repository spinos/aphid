/*
 *  HullContainer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "shapeDrawer.h"
#include <Polytode.h>

class HullContainer : public Polytode {
public:
	HullContainer();
	virtual ~HullContainer();

	void initHull();
	void renderWorld(ShapeDrawer * drawer);

	void processHull();
	char searchVisibleFaces(Vertex *v);
	char searchHorizons();
	char spawn(Vertex *v);
	char finishStep(Vertex *v);
	void addConflict(Facet *f, Vertex *v);
	void addConflict(Facet *f, Facet *a, Facet *b);
	void removeConflict(Facet *f);
	
protected:
	std::vector<Facet *>visibleFaces;
	Edge * m_horizon;
	int m_currentVertexId;
	int m_numHorizon;
};
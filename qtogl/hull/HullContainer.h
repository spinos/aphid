/*
 *  HullContainer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef HULL_CONTAINER_H
#define HULL_CONTAINER_H

#include <topo/Polytope.h>

namespace aphid {

class Vertex;
class Facet;
class Edge;

}

class HullContainer : public aphid::Polytope {
public:
	HullContainer();
	virtual ~HullContainer();

	void initHull();
	//void renderWorld(ShapeDrawer * drawer);

	void processHull();
	char searchVisibleFaces(aphid::Vertex *v);
	char searchHorizons();
	char spawn(aphid::Vertex *v);
	char finishStep(aphid::Vertex *v);
	void addConflict(aphid::Facet *f, aphid::Vertex *v);
	void addConflict(aphid::Facet *f, aphid::Facet *a, aphid::Facet *b);
	void removeConflict(aphid::Facet *f);
	
protected:
	std::vector<aphid::Facet *>visibleFaces;
	aphid::Edge * m_horizon;
	int m_currentVertexId;
	int m_numHorizon;
};
#endif

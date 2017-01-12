/*
 *  Polytope.h
 *  convexHull
 *
 *  Created by jian zhang on 9/10/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TOPO_POLYTOPE_H
#define APH_TOPO_POLYTOPE_H

#include <vector>

namespace aphid {

class Vertex;
class Facet;

class Polytope {

	std::vector<Vertex *>m_vertices;
	std::vector<Facet *>m_faces;
	
public:
	Polytope();
	virtual ~Polytope();
	
	void destroy();
	
	int getNumVertex() const;
	int getNumFace() const;
	
	void addVertex(Vertex * p);
	const Vertex & getVertex(int idx) const;
	Vertex * vertex(int idx);
	
	void addFacet(Facet * f);
	const Facet & getFacet(int idx) const;
	void removeFaces();
	
protected:
	std::vector<Facet *> & faces();
	
};

}
#endif

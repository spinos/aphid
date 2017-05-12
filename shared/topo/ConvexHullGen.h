/*
 *  ConvexHullGen.h
 *  hull
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_CONVECT_HULL_GEN_H
#define APH_CONVECT_HULL_GEN_H

#include <topo/Polytope.h>

namespace aphid {

class Vector3F;
class Vertex;
class Facet;
class Edge;
class ConflictGraph;
class GraphArch;

class ATriangleMesh;

class ConvexHullGen : public Polytope {
	
	std::vector<Facet *>visibleFaces;
	std::vector<ConflictGraph *> m_conflg;
	std::vector<GraphArch *> m_arch;
	Edge * m_horizon;
	int m_currentVertexId;
	int m_numHorizon;
	
public:
	ConvexHullGen();
	virtual ~ConvexHullGen();

	void addSample(const Vector3F & p);
	void processHull();
	void extractMesh(ATriangleMesh * msh);
	void extractMesh(Vector3F* pos, Vector3F* nml, unsigned* inds, int offset);

protected:
	Vector3F getCenter() const;
	char searchVisibleFaces(Vertex *v);
	char searchHorizons();
	char spawn(Vertex *v);
	char finishStep(Vertex *v);
	void addConflict(Facet *f, Vertex *v);
	void addConflict(Facet *f, Facet *a, Facet *b);
	void removeConflict(Facet *f);
	void checkFaceNormal(Vector3F* pos, Vector3F* nml,
			const Vector3F & vref) const;
	
private:
	
};

}
#endif

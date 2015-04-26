/*
 *  BccMesh.h
 *  testbcc
 *
 *  Created by jian zhang on 4/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <ATetrahedronMesh.h>
class GeometryArray;
class KdIntersection;
class BccGrid;

class BccMesh : public ATetrahedronMesh {
public:
	BccMesh();
	virtual ~BccMesh();
	
	void create(GeometryArray * geoa, KdIntersection * anchorIntersect, int level);
protected:

private:
	void resetAnchors(unsigned n);
private:
	BaseBuffer * m_anchors;
	KdIntersection * m_intersect;
	BccGrid * m_grid;
};
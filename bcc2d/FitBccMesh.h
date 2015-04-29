/*
 *  FitBccMesh.h
 *  testbcc
 *
 *  Created by jian zhang on 4/27/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <ATetrahedronMesh.h>
class GeometryArray;
class KdIntersection;

class FitBccMesh : public ATetrahedronMesh {
public:
	FitBccMesh();
	virtual ~FitBccMesh();
	
	void create(GeometryArray * geoa, KdIntersection * anchorPoints,
					float groupNCvRatio,
	           unsigned minNumGroups,
	           unsigned maxNumGroups);
protected:

private:
	void addAnchors(KdIntersection * anchorIntersect);
private:
	
};
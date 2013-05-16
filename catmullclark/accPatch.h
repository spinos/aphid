/*
 *  accPatch.h
 *  catmullclark
 *
 *  Created by jian zhang on 10/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <bezierPatch.h>
class PatchTopology;
class AccStencil;
class AccPatch : public BezierPatch {
public:
	AccPatch();
	~AccPatch();
	void evaluateContolPoints(PatchTopology& topo);
	void processCornerControlPoints(PatchTopology& topo);
	void processEdgeControlPoints(PatchTopology& topo);
	void processInteriorControlPoints(PatchTopology& topo);
	static AccStencil* stencil;
};

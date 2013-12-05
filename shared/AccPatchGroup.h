/*
 *  AccPatchGroup.h
 *  aphid
 *
 *  Created by jian zhang on 11/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <AllMath.h>
class AccPatch;
class BezierPatchHirarchy;
class IntersectionContext;
class AccPatchGroup {
public:
	AccPatchGroup();
	virtual ~AccPatchGroup();
	void createAccPatches(unsigned n);
	AccPatch* beziers() const;
	BezierPatchHirarchy * hirarchies() const;
	void createBezierHirarchy(unsigned idx);

    void recursiveBezierClosestPoint1(IntersectionContext * ctx, int level, unsigned current) const;
		
private:
	AccPatch* m_bezier;
	BezierPatchHirarchy * m_hirarchy;
};
#ifndef RANDOMCURVE_H
#define RANDOMCURVE_H

/*
 *  RandomCurve.h
 *  testkdtree
 *
 *  Created by jian zhang on 4/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
class GeometryArray;
class BezierPatch;
class RandomCurve {
public:
	RandomCurve();
	void create(GeometryArray * result, 
				unsigned nu, unsigned nv, 
				BezierPatch * base,
				const Vector3F & generalDir,
				int sn, int sm,
				float fr);
};
#endif        //  #ifndef RANDOMCURVE_H

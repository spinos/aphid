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
class RandomCurve {
public:
	RandomCurve();
	void create(GeometryArray * result, unsigned n, 
				const Vector3F & base,
				const Vector3F & translation,
				const Vector3F & generalDir,
				float radius);
};
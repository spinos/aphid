/*
 *  HermiteCurve.h
 *  softIk
 *
 *  Created by jian zhang on 3/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
class HermiteCurve {
public:
	HermiteCurve();
	virtual ~HermiteCurve();
	
	Vector3F interpolate(const float & s) const; // s between 0 and 1

	Vector3F _P[2], _T[2];
private:
};
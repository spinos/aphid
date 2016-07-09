/*
 *  Convexity.cpp
 *  
 *
 *  Created by jian zhang on 7/8/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Convexity.h"
#include "tetrahedron_math.h"

using namespace aphid;

namespace ttg {

Convexity::Convexity() {}

bool Convexity::CheckDistanceFourPoints(const aphid::Vector3F & a,
						const aphid::Vector3F & b,
						const aphid::Vector3F & c,
						const aphid::Vector3F & d,
						const aphid::Vector3F & p0,
						const float & r)
{
	if(p0.distanceTo(a) < r )
		return false;
		
	if(p0.distanceTo(b) < r )
		return false;
		
	if(p0.distanceTo(c) < r )
		return false;
		
	if(p0.distanceTo(d) < r )
		return false;
		
	return true;
}

/// close to or intersect either plane
bool Convexity::CheckDistanceTwoPlanes(const aphid::Vector3F & a,
						const aphid::Vector3F & b,
						const aphid::Vector3F & c,
						const aphid::Vector3F & d,
						const aphid::Vector3F & p0,
						const float & r)
{
	Plane plane1(a, b, c);
		
	float d0 = plane1.distanceTo(p0);
	if(Absolute<float>(d0) < r)
		return false;
			
	float d1 = plane1.distanceTo(d);
	if(d1 * d0 < 0.f)
		return false;
		
	Plane plane2(a, b, d);
		
	d0 = plane2.distanceTo(p0);
	if(Absolute<float>(d0) < r)
		return false;
			
	d1 = plane2.distanceTo(c);
	if(d1 * d0 < 0.f)
		return false;

	return true;
}

bool Convexity::CheckTetraVolume(const aphid::Vector3F & a,
						const aphid::Vector3F & b,
						const aphid::Vector3F & c,
						const aphid::Vector3F & d)
{
	return tetrahedronVolume1(a, b, c, d) > 1e-3f;
}

bool Convexity::CheckDistancePlane(const aphid::Vector3F & a,
						const aphid::Vector3F & b,
						const aphid::Vector3F & c,
						const aphid::Vector3F & d,
						const aphid::Vector3F & p0,
						const float & r)
{
	Plane plane1(a, b, c, d);
	
	return Absolute<float>(plane1.distanceTo(p0) ) > r;
}

}
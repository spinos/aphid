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
	return tetrahedronVolume1(a, b, c, d) > 1e-2f;
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

bool Convexity::CheckDistancePlane(const aphid::Vector3F & a,
						const aphid::Vector3F & b,
						const aphid::Vector3F & c,
						const aphid::Vector3F & p0,
						const float & r)
{
	Plane plane1(a, b, c);
	
	return Absolute<float>(plane1.distanceTo(p0) ) > r;
}

bool Convexity::CheckInsideTetra(const aphid::Vector3F & a,
						const aphid::Vector3F & b,
						const aphid::Vector3F & c,
						const aphid::Vector3F & d,
						const aphid::Vector3F & p0,
						const float & r)
{
	if(!CheckDistanceFourPoints(a, b, c, d, p0, r) )
		return false;
		
	bool reversed = tetrahedronVolume1(a, b, c, d) < 0.f;
	
	Vector3F v[4];
	v[0] = a;
	v[1] = b;
	if(reversed) {
		v[2] = d;
		v[3] = c;
	}
	else {
		v[2] = c;
		v[3] = d;
	}
	
	if(!pointInsideTetrahedronTest(p0, v) )
		return false;
		
	float r2 = r * .33f;
	
	if(!CheckDistancePlane(v[0], v[1], v[2], p0, r2) )
		return false;
		
	if(!CheckDistancePlane(v[0], v[3], v[1], p0, r2) )
		return false;
		
	if(!CheckDistancePlane(v[0], v[2], v[3], p0, r2) )
		return false;
		
	if(!CheckDistancePlane(v[1], v[3], v[2], p0, r2) )
		return false;
		
	return true;
}

bool Convexity::CheckInsideTetra(const aphid::Vector3F * v,
						const aphid::Vector3F & p0)
{
	Float4 coord;
	if(!pointInsideTetrahedronTest1(p0, v, &coord) )
		return false;
		
	float mnc = coord.x;
	if(mnc > coord.y)
		mnc = coord.y;
		
	if(mnc > coord.z)
		mnc = coord.z;
		
	if(mnc > coord.w)
		mnc = coord.w;
		
	float mxc = coord.x;
	if(mxc < coord.y)
		mxc = coord.y;
		
	if(mxc < coord.z)
		mxc = coord.z;
		
	if(mxc < coord.w)
		mxc = coord.w;	

/// not on edge or face	or vertex
	return mnc > 0.02f && mxc < .98f;
}

}
/*
 *  FeatherDeformParam.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherDeformParam.h"
#include <math/Matrix33F.h>
#include <gpr/GPInterpolate.h>

using namespace aphid;

FeatherDeformParam::FeatherDeformParam()
{}

FeatherDeformParam::~FeatherDeformParam()
{}

void FeatherDeformParam::predictRotation(aphid::Matrix33F & dst,
						const float * x,
						const float & relspeed)
{
	sideInterp()->predict(x);
	upInterp()->predict(x);
	
	const float * sideY = sideInterp()->predictedY().column(0);
	const float * upY = upInterp()->predictedY().column(0);
	
	Vector3F vside(sideY[0], sideY[1], sideY[2]);
	vside.normalize();
	Vector3F vup(upY[0], upY[1], upY[2]);
	vup.normalize();
	
	vside = Vector3F::XAxis + (vside - Vector3F::XAxis) * relspeed;
	vside.normalize();
	
	vup = Vector3F::YAxis + (vup - Vector3F::YAxis) * relspeed;
	vup.normalize();
	
	Vector3F vfront = vside.cross(vup);
	vside = vup.cross(vfront);
	vside.normalize();
	
	dst.fill(vside, vup, vfront);
}

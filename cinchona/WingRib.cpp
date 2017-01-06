/*
 *  WingRib.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "WingRib.h"

using namespace aphid;

WingRib::WingRib(const float & c,
			const float & m,
			const float & p,
			const float & t) : Airfoil(c, m, p, t)
{}

WingRib::~WingRib()
{}

void WingRib::getPoint(Vector3F & dst, const float & param) const
{
	float x = param;
	if(param < 0.f) {
		x = 1.f + param;
	}
	
	float yc = calcYc(x);
	float yt = calcYt(x);
	if(param < 0.f) {
		yt = -yt;
	}
	dst.set(x*chord(), yc+yt, 0.f);
	dst = transform(dst);
	
}


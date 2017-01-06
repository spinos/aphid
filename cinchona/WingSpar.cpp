/*
 *  WingSpar.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "WingSpar.h"
#include <math/Vector3F.h>

using namespace aphid;

WingSpar::WingSpar(const int & np) : 
HermiteInterpolatePiecewise<float, Vector3F >(np)
{}

WingSpar::~WingSpar()
{}

void WingSpar::setKnot(const int & i,
				const Vector3F & p,
				const Vector3F & t)
{
	if(i==0) {
		setPieceBegin(i, p, t);
	} else if(i==numPieces() ) {
		setPieceEnd(i-1, p, t);
	} else {
		setPieceEnd(i-1, p, t);
		setPieceBegin(i, p, t);
	}
}

void WingSpar::getPoint(Vector3F & dst, const int & i) const
{
	int seg = i/25;
	float x = (float)(i-seg*25) / 25.f;
	dst = interpolate(seg, x);
}
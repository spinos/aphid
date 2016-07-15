/*
 *  ADistanceField.cpp
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ADistanceField.h"

namespace aphid {

ADistanceField::ADistanceField()
{}

ADistanceField::~ADistanceField()
{}

void ADistanceField::nodeColor(Vector3F & dst, const DistanceNode & n,
						const float & scale) const
{
	if(n.label == sdf::StBackGround)
		return dst.set(.3f, .3f, .3f);
	
	if(n.val > 0.f) {
		float r = MixClamp01F<float>(1.f, 0.f, n.val * scale);
		dst.set(r, 1.f - r, 0.f);
	}
	else {
		float b = MixClamp01F<float>(1.f, 0.f, -n.val * scale);
		dst.set(.5f * (1.f - b), 1.f - b, b);
	}
}

void ADistanceField::initNodes()
{
	const int n = numNodes();
	int i = 0;
	for(;i<n;++i) {
		DistanceNode & d = nodes()[i];
		d.label = sdf::StBackGround;
		d.val = 1e9f;
	}
}

}
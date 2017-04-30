/*
 *  RandomCurve.cpp
 *  testkdtree
 *
 *  Created by jian zhang on 4/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "RandomCurve.h"
#include <BezierCurve.h>
#include <CurveBuilder.h>
#include <GeometryArray.h>
#include <BezierPatch.h>
namespace aphid {

RandomCurve::RandomCurve() {}

void RandomCurve::create(GeometryArray * result, 
				unsigned nu, unsigned nv,
				BezierPatch * base,
				const Vector3F & generalDir,
				int sn, int sm,
				float fr)
{
	unsigned i, j, k, s;
	const float dirl = generalDir.length();
	CurveBuilder cb;
	float du = 1.f/(float)nu;
	float dv = 1.f/(float)nv;
	float u, v;
	Vector3F p, vel, gra;
	for(j = 0; j < nv; j++) {
	for(i = 0; i < nu; i++) {
		BezierCurve * c = new BezierCurve;
		
		u = du * (.5f + i + RandomFn11() * .4f);
		v = dv * (.5f + j + RandomFn11() * .4f);
		
		base->evaluateSurfacePosition(u, v, &p);
		
		cb.addVertex(p);
		
		s = sn + (sm - sn) * RandomF01();
		vel = generalDir + Vector3F(RandomFn11(), RandomFn11(), RandomFn11()) * dirl * fr * .5f;
		vel.normalize();
		vel *= dirl;
		
		for(k=1; k< s; k++) {
			p += vel;
			cb.addVertex(p);
			
			gra = Vector3F(RandomFn11(), RandomFn11(), RandomFn11());
			gra.normalize();
			gra *= dirl * fr * .1f;
		
			vel += gra;
		}
		
		cb.finishBuild(c);
		result->setGeometry(c, j * nu + i);
	}
	}
}

}
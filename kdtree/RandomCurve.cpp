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

RandomCurve::RandomCurve() {}

void RandomCurve::create(GeometryArray * result, unsigned n, 
				const Vector3F & base,
				const Vector3F & translation,
				const Vector3F & generalDir,
				float radius)
{
	unsigned i, j, nv;
	const Vector3F deltaTranslation = translation * (1.f/(float)n);
	const float dirl = generalDir.length();
	CurveBuilder cb;
	Vector3F curBase = base;
	Vector3F p, vel, gra;
	for(i = 0; i < n; i++) {
		BezierCurve * c = new BezierCurve;
		
		p = curBase + Vector3F( RandomFn11() * radius, RandomFn11() * radius * .02f,  RandomFn11() * radius);
		cb.addVertex(p);
		
		nv = 10 + 3 * RandomF01();
		vel = generalDir + Vector3F(RandomFn11(), RandomFn11(), RandomFn11()) * dirl * .22f;
		vel.normalize();
		vel *= dirl;
		
		for(j=1; j< nv; j++) {
			p += vel;
			cb.addVertex(p);
			
			gra = Vector3F(RandomFn11(), RandomFn11(), RandomFn11());
			gra.normalize();
			gra *= dirl * .12f;
		
			vel += gra;
		}
		
		cb.finishBuild(c);
		result->setGeometry(c, i);
		curBase += deltaTranslation;
	}
}
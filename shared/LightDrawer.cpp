/*
 *  LightDrawer.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "LightDrawer.h"
#include "LightGroup.h"
#include "DistantLight.h"
#include "PointLight.h"
#include "SquareLight.h"
LightDrawer::LightDrawer() {}
LightDrawer::~LightDrawer() {}

void LightDrawer::drawLights(const LightGroup & grp) const
{
	const unsigned nl = grp.numLights();
	for(unsigned i = 0; i < nl; i++) {
		drawLight(grp.getLight(i));
	}
}

void LightDrawer::drawLight(BaseLight * l) const
{
	transform(l);
	useSolid();
	useColor(l->lightColor());
	switch (l->entityType()) {
		case TypedEntity::TDistantLight:
			drawDistantLight(static_cast<DistantLight *>(l));
			break;
		case TypedEntity::TPointLight:
			drawPointLight(static_cast<PointLight *>(l));
			break;
		case TypedEntity::TSquareLight:
			drawSquareLight(static_cast<SquareLight *>(l));
			break;
		default:
			break;
	}
}

void LightDrawer::drawDistantLight(DistantLight * l) const
{
	glPushMatrix();
	useSpace(l->space());
	drawDisc(1.f);
	Float3 dark(0.1f, 0.23f, 0.34f);
	useColor(dark);
	Vector3F a(0.f, 0.f, 16.f);
	Vector3F b(0.f, 0.f, -16.f);
	arrow(a, b);
	glPopMatrix();
}

void LightDrawer::drawPointLight(PointLight * l) const
{
	glPushMatrix();
	useSpace(l->space());
	sphere(1.f);
	glPopMatrix();
}

void LightDrawer::drawSquareLight(SquareLight * l) const
{
	glPushMatrix();
	useSpace(l->space());
	drawSquare(l->square());
	glPopMatrix();
}

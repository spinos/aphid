/*
 *  MlDrawer.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlDrawer.h"
#include <AccPatchMesh.h>
#include <PointInsidePolygonTest.h>
MlDrawer::MlDrawer() {}
MlDrawer::~MlDrawer() {}

void MlDrawer::drawFeather(MlSkin * skin) const
{
	const unsigned nf = skin->numFeathers();
	if(nf < 1) return;
	useDepthTest(0);
	setColor(1.f, 0.f, 0.f);
	AccPatchMesh * mesh = skin->bodyMesh();
	Vector3F p;
	for(unsigned i = 0; i < nf; i++) {
		MlCalamus * c = skin->getCalamus(i);
		mesh->pointOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
		Matrix33F frm = mesh->tangentFrame(c->faceIdx(), c->patchU(), c->patchV());
		
		Matrix33F space;
		space.rotateX(c->rotateX());
		space.multiply(frm);
		
		Matrix33F ys;
		ys.rotateY(c->rotateY());
		ys.multiply(space);
		
		Vector3F d(0.f, 0.f, c->scale());
		d = ys.transform(d);
		d = p + d;
		arrow(p, d);
	}
	
	drawActiveFeather(skin);
}

void MlDrawer::drawActiveFeather(MlSkin * skin) const
{
	const unsigned num = skin->numActiveFeather();
	if(num < 1) return;
	
	AccPatchMesh * mesh = skin->bodyMesh();
	Vector3F p;
	unsigned i;
	for(i = 0; i < num; i++) {
		MlCalamus * c = skin->getActive(i);
		mesh->pointOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
		Matrix33F frm = mesh->tangentFrame(c->faceIdx(), c->patchU(), c->patchV());
		
		Matrix33F space;
		space.rotateX(c->rotateX());
		
		space.multiply(frm);
		
		Matrix33F ys;
		ys.rotateY(c->rotateY());
		ys.multiply(space);
		
		Vector3F d(0.f, 0.f, c->scale());
		d = ys.transform(d);
		d = p + d;
		arrow(p, d);
	}
}

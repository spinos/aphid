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
MlDrawer::MlDrawer() {}
MlDrawer::~MlDrawer() {}

void MlDrawer::drawFeather(MlSkin * skin)
{//printf("hit\n");
	const unsigned nf = skin->numFeathers();
	if(nf < 1) return;
	useDepthTest(0);
	beginPoint(5.f);
	setColor(1.f, 0.f, 0.f);
	AccPatchMesh * mesh = skin->bodyMesh();
	Vector3F p;
	for(unsigned i = 0; i < nf; i++) {
		MlCalamus * c = skin->getCalamus(i);
		mesh->pointOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
		//printf("c %i %f %f ", c->faceIdx(), c->patchU(), c->patchV());
		//printf("p %f %f %f ", p.x, p.y, p.z);
		vertex(p);
	}
	end();
}
/*
 *  MlTessellate.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlTessellate.h"
#include <MeshTopology.h>
#include <AccPatchMesh.h>
#include <MlFeather.h>
MlTessellate::MlTessellate() : m_numSegment(0) {}

MlTessellate::~MlTessellate() {}

void MlTessellate::cleanup()
{
	m_numSegment = 0;
	BaseTessellator::cleanup();
}

void MlTessellate::setFeather(MlFeather * feather)
{
	if((unsigned)feather->numSegment() == m_numSegment) {
		return;
	}
	else if((unsigned)feather->numSegment() < m_numSegment) {
		m_numSegment = feather->numSegment();
		setNumVertices((m_numSegment + 1) * 3);
		setNumIndices(m_numSegment * 2 * 4);
		return;
	}
	
	cleanup();

	m_numSegment = feather->numSegment();
	
	create((m_numSegment + 1) * 3, m_numSegment * 2 * 4);
	
	unsigned * ind = indices();
	unsigned curF = 0;
	for(short i = 0; i < feather->numSegment(); i++) {
		ind[curF] = 3 * i;
		ind[curF + 1] = 3 * i + 1;
		ind[curF + 2] = 3 * (i + 1) + 1;
		ind[curF + 3] = 3 * (i + 1);
		curF += 4;
		ind[curF] = 3 * i;
		ind[curF + 1] = 3 * (i + 1);
		ind[curF + 2] = 3 * (i + 1) + 2;
		ind[curF + 3] = 3 * i + 2;
		curF += 4;
	}
}

void MlTessellate::evaluate(MlFeather * feather)
{
	Vector3F * cvs = vertices();
	Vector3F * nors = normals(); 
	Vector2F * uvs = texcoords();
	unsigned curF = 0;
	Vector3F uvc;
	for(short i = 0; i <= feather->numSegment(); i++) {
		cvs[curF] = *feather->patchCenterP(i);
		cvs[curF + 1] = *feather->patchWingP(i, 0);
		cvs[curF + 2] = *feather->patchWingP(i, 1);
		uvc = *feather->patchCenterUV(i);
		uvs[curF].set(uvc.x, uvc.y);
		uvc = *feather->patchWingUV(i, 0);
		uvs[curF + 1].set(uvc.x, uvc.y); 
		uvc = *feather->patchWingUV(i, 1);
		uvs[curF + 2].set(uvc.x, uvc.y);
		curF += 3;
	}
	curF = 0;
	Vector3F N, deviateL, deviateR;
	for(short i = 0; i < feather->numSegment(); i++) {
		N = *feather->normal(i);
		deviateR = cvs[curF + 1] - cvs[curF];
		deviateL = cvs[curF + 2] - cvs[curF];
		nors[curF] = (N + deviateR * 0.7f + deviateL * 0.7f).normal();
		
		deviateR = deviateR.cross(N);
		deviateR.normalize();
		nors[curF+1] = (N + deviateR * 0.23f).normal();
		
		deviateL = deviateL.cross(N);
		deviateL.normalize();
		nors[curF+2] = (N + deviateL * 0.23f).normal();
		curF += 3;
	}
	nors[curF] = nors[curF - 3];
	nors[curF+1] = nors[curF - 2];
	nors[curF+2] = nors[curF - 1];
}

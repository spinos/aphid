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

void MlTessellate::setFeather(MlFeather * feather)
{
	if((unsigned)feather->numSegment() == m_numSegment) return;
	cleanup();
	
	createVertices(feather);
	createIndices(feather);
	m_numSegment = feather->numSegment();
}

void MlTessellate::createVertices(MlFeather * feather)
{
	m_numVertices = (feather->numSegment() + 1) * 3;
	m_cvs = new Vector3F[m_numVertices];
	m_normals = new Vector3F[m_numVertices];
	m_uvs = new Vector2F[m_numVertices];
}

void MlTessellate::createIndices(MlFeather * feather)
{
	m_numIndices = feather->numSegment() * 2 * 4;
	m_indices = new unsigned[m_numIndices];
	
	unsigned curF = 0;
	for(short i = 0; i < feather->numSegment(); i++) {
		m_indices[curF] = 3 * i;
		m_indices[curF + 1] = 3 * i + 1;
		m_indices[curF + 2] = 3 * (i + 1) + 1;
		m_indices[curF + 3] = 3 * (i + 1);
		curF += 4;
		m_indices[curF] = 3 * i;
		m_indices[curF + 1] = 3 * (i + 1);
		m_indices[curF + 2] = 3 * (i + 1) + 2;
		m_indices[curF + 3] = 3 * i + 2;
		curF += 4;
	}
}

void MlTessellate::evaluate(MlFeather * feather)
{
	unsigned curF = 0;
	Vector3F uvc;
	for(short i = 0; i <= feather->numSegment(); i++) {
		m_cvs[curF] = *feather->patchCenterP(i);
		m_cvs[curF + 1] = *feather->patchWingP(i, 0);
		m_cvs[curF + 2] = *feather->patchWingP(i, 1);
		uvc = *feather->patchCenterUV(i);
		m_uvs[curF].set(uvc.x, uvc.y);
		uvc = *feather->patchWingUV(i, 0);
		m_uvs[curF + 1].set(uvc.x, uvc.y); 
		uvc = *feather->patchWingUV(i, 1);
		m_uvs[curF + 2].set(uvc.x, uvc.y);
		curF += 3;
	}
	curF = 0;
	Vector3F N;
	for(short i = 0; i < feather->numSegment(); i++) {
		N = *feather->normal(i);
		m_normals[curF] = N;
		m_normals[curF+1] = N;
		m_normals[curF+2] = N;
		curF += 3;
	}
	m_normals[curF] = N;
	m_normals[curF+1] = N;
	m_normals[curF+2] = N;
}

/*
 *  LODQuad.cpp
 *  easymodel
 *
 *  Created by jian zhang on 11/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "Vector3F.h"
#include "Vector2F.h"
#include "LODQuad.h"

LODQuad::LODQuad() {}
LODQuad::~LODQuad() {}

void LODQuad::setCorner(Vector3F p, int i)
{
	_corners[i] = p;
}

Vector3F LODQuad::getCorner(int i) const
{
	return _corners[i];
}

void LODQuad::setDetail(float d, int i)
{
	_details[i] = d;
}

float LODQuad::getDetail(int i) const
{
	return _details[i];
}

void LODQuad::evaluateSurfaceLOD(float u, float v, float * detail) const
{
	Vector2F L0(1-u,1-v);
	Vector2F L1(u,v);

	float d = 
		(_details[0] * L0.x + _details[1] * L1.x) * L0.y + 
		(_details[2] * L0.x + _details[3] * L1.x) * L1.y;
	*detail = d;
}

float LODQuad::getMaxLOD() const
{
	float max = _details[0];
	if(_details[1] > max) max = _details[1];
	if(_details[2] > max) max = _details[2];
	if(_details[3] > max) max = _details[3];
	
	return max;
}

float LODQuad::getMaxEdgeLength() const
{
	Vector3F ab = _corners[3] - _corners[0];
	Vector3F cd = _corners[1] - _corners[2];
	float lab = ab.length();
	float lcd = cd.length();
	if(lab > lcd) return lab;
	return lcd;
}

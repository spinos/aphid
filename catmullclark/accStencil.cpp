/*
 *  accStencil.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <Vector3F.h>
#include "accStencil.h"
#include <iostream>

AccStencil::AccStencil() {}
AccStencil::~AccStencil() {}

void AccStencil::setVertexPosition(Vector3F* data)
{
	_positions = data;
}

void AccStencil::setVertexNormal(Vector3F* data)
{
	_normals = data;
}

Vector3F AccStencil::computePositionOnCornerOnBoundary() const
{
	return _positions[centerIndex] * (2.f / 3.f) +  _positions[edgeIndices[0]] * (1.f / 6.f) +  _positions[edgeIndices[1]] * (1.f / 6.f);
}

Vector3F AccStencil::computePositionOnCorner() const
{
	float n = valence * valence;
	float e = 4.f;
	float c = 1.f;
	float sum = n;
	Vector3F res = _positions[centerIndex] * n;
	for(int i = 0; i < valence; i++)
	{
		res += _positions[edgeIndices[i]] * e;
		res += _positions[cornerIndices[i]] * c;
		sum += e;
		sum += c;
	}
	return res / sum;
}

Vector3F AccStencil::computePositionOnEdgeOnBoundary() const
{
	return _positions[edgeIndices[0]] * (2.f / 3.f) + _positions[edgeIndices[1]] * (1.f / 3.f);
}

Vector3F AccStencil::computePositionOnEdge() const
{
	Vector3F res = _positions[edgeIndices[0]] * 2.f * valence;
	res += _positions[edgeIndices[1]] * 4.f;
	res += _positions[cornerIndices[0]] * 2.f;
	res += _positions[cornerIndices[1]] * 2.f;
	res += _positions[cornerIndices[2]];
	res += _positions[cornerIndices[3]];
	return res / (2.f * valence + 10.f);
}

Vector3F AccStencil::computePositionInterior() const
{
	Vector3F res = _positions[cornerIndices[0]] * valence;
	res += _positions[cornerIndices[1]] * 2.f;
	res += _positions[cornerIndices[2]] * 2.f;
	res += _positions[cornerIndices[3]];
	return res / (valence + 5.f);
}

Vector3F AccStencil::computeNormalOnCornerOnBoundary() const
{
	return _normals[centerIndex] * (2.f / 3.f) +  _normals[edgeIndices[0]] * (1.f / 6.f) +  _normals[edgeIndices[1]] * (1.f / 6.f);
}

Vector3F AccStencil::computeNormalOnCorner() const
{
	float n = valence * valence;
	float e = 4.f;
	float c = 1.f;
	float sum = n;
	Vector3F res = _normals[centerIndex] * n;
	for(int i = 0; i < valence; i++)
	{
		res += _normals[edgeIndices[i]] * e;
		res += _normals[cornerIndices[i]] * c;
		sum += e;
		sum += c;
	}
	return res / sum;
}

Vector3F AccStencil::computeNormalOnEdgeOnBoundary() const
{
	return _normals[edgeIndices[0]] * (2.f / 3.f) + _normals[edgeIndices[1]] * (1.f / 3.f);
}

Vector3F AccStencil::computeNormalOnEdge() const
{
	Vector3F res = _normals[edgeIndices[0]] * 2.f * valence;
	res += _normals[edgeIndices[1]] * 4.f;
	res += _normals[cornerIndices[0]] * 2.f;
	res += _normals[cornerIndices[1]] * 2.f;
	res += _normals[cornerIndices[2]];
	res += _normals[cornerIndices[3]];
	return res / (2.f * valence + 10.f);
}

Vector3F AccStencil::computeNormalInterior() const
{
	Vector3F res = _normals[cornerIndices[0]] * valence;
	res += _normals[cornerIndices[1]] * 2.f;
	res += _normals[cornerIndices[2]] * 2.f;
	res += _normals[cornerIndices[3]];
	return res / (valence + 5.f);
}

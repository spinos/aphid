/*
 *  tessellator.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 11/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <AllMath.h>
#include "tessellator.h"

Tessellator::Tessellator() : _positions(0), _normals(0), _vertices(0)//, _displacementMap(0) 
{}
Tessellator::~Tessellator() 
{
	cleanup();
}

void Tessellator::cleanup()
{
	if(_positions) delete[] _positions;
	if(_normals) delete[] _normals;
	if(_texcoords) delete[] _texcoords;
	if(_vertices) delete[] _vertices;
}

void Tessellator::setNumSeg(int n)
{
	cleanup();
	_numSeg = n;
	_positions = new Vector3F[numVertices()];
	_normals = new Vector3F[numVertices()];
	_texcoords = new Vector3F[numVertices()];
	_vertices = new int[numIndices()];
}
/*
void Tessellator::setDisplacementMap(ZEXRImage* image)
{
	_displacementMap = image;
}
*/
void Tessellator::evaluate(BezierPatch& bezier)
{
	float delta = 1.f / _numSeg;
	for(int j=0; j < _numSeg + 1; j++)
	{
		for(int i = 0; i < _numSeg + 1; i++)
		{
			int idx = j * (_numSeg + 1) + i;
			bezier.evaluateSurfaceNormal(delta * i, delta * j, &_normals[idx]);
			bezier.evaluateSurfaceTexcoord(delta * i, delta * j, &_texcoords[idx]);
			bezier.evaluateSurfacePosition(delta * i, delta * j, &_positions[idx]);
		}
	}
	
	//if(_displacementMap)
	//{
	//	displacePositions(bezier);
	//	calculateNormals();
	//}
	
	//testLOD(bezier);
	
	for(int j=0; j < _numSeg; j++)
	{
		for(int i = 0; i < _numSeg; i++)
		{
			_vertices[(j * _numSeg + i) * 4] = j * (_numSeg + 1) + i;
			_vertices[(j * _numSeg + i) * 4 + 1] = j * (_numSeg + 1) + i + 1;
			_vertices[(j * _numSeg + i) * 4 + 2] = (j + 1) * (_numSeg + 1) + i + 1;
			_vertices[(j * _numSeg + i) * 4 + 3] = (j + 1) * (_numSeg + 1) + i;
		}
	}
}

void Tessellator::displacePositions(BezierPatch& bezier)
{
	float le = 4.f;
	float delta = 1.f / _numSeg;
	float d[1];
	for(int j=0; j < _numSeg + 1; j++)
	{
		for(int i = 0; i < _numSeg + 1; i++)
		{
			//bezier.evaluateSurfaceLOD(delta * i, delta * j, &d);
			int idx = j * (_numSeg + 1) + i;
			//_displacementMap->sample(_texcoords[idx].x, _texcoords[idx].y, log2f(le) + 4, 1, d);
			_positions[idx] += _normals[idx] * (d[0] - 0.5f);
			
		}
	}
}

void Tessellator::testLOD(BezierPatch& bezier)
{
	float le = 4.f;
	float delta = 1.f / _numSeg;
	float d[1];
	for(int j=0; j < _numSeg + 1; j++)
	{
		for(int i = 0; i < _numSeg + 1; i++)
		{
			//bezier.evaluateSurfaceLOD(delta * i, delta * j, &d);
			int idx = j * (_numSeg + 1) + i;
			//_displacementMap->sample(_texcoords[idx].x, _texcoords[idx].y, log2f(le) + 4, 1, d);// + bezier.getLODBase());
			_normals[idx].x = d[0];
			_normals[idx].y = d[0];
			_normals[idx].z = d[0];
		}
	}
}

void Tessellator::calculateNormals()
{
	Vector3F dpdu, dpdv;
	for(int j=0; j < _numSeg + 1; j++)
	{
		for(int i = 0; i < _numSeg + 1; i++)
		{
			if(i != _numSeg)
				dpdu = *p(i + 1 , j) - *p(i, j);
			else
				dpdu = *p(i , j) - *p(i - 1, j);
			
			if(j != _numSeg)
				dpdv = *p(i, j + 1) - *p(i, j);
			else
				dpdv = *p(i, j) - *p(i, j - 1);
				
			*n(i , j) = dpdv.cross(dpdu).normal();
		}
	}
}

int* Tessellator::getVertices() const
{
	return _vertices;
}

float* Tessellator::getPositions() const
{
	return (float*)_positions;
}

float* Tessellator::getNormals() const
{
	return (float*)_normals;
}

Vector3F* Tessellator::p(int i, int j)
{
	return &_positions[j * (_numSeg + 1) + i];
}
	
Vector3F* Tessellator::n(int i, int j)
{
	return &_normals[j * (_numSeg + 1) + i];
}

int Tessellator::numVertices() const
{
	return (_numSeg + 1) * (_numSeg + 1);
}

int Tessellator::numIndices() const
{
	return _numSeg * _numSeg * 4;
}

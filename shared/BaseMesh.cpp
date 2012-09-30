/*
 *  BaseMesh.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>

#include "BaseMesh.h"

BaseMesh::BaseMesh() : _vertices(0), _indices(0) {}
BaseMesh::~BaseMesh()
{
	if(_vertices) delete[] _vertices;
	if(_indices) delete[] _indices;
}

void BaseMesh::createVertices(unsigned num)
{
	_vertices = new Vector3F[num];
	_numVertices = num;
}

void BaseMesh::createIndices(unsigned num)
{
	_indices = new unsigned[num];
	_numFaceVertices = num;
}



Vector3F * BaseMesh::vertices()
{
	return _vertices;
}

unsigned * BaseMesh::indices()
{
	return _indices;
}

unsigned BaseMesh::getNumVertices() const
{
	return _numVertices;
}

unsigned BaseMesh::getNumFaceVertices() const
{
	return _numFaceVertices;
}

Vector3F * BaseMesh::getVertices() const
{
	return _vertices;
}

unsigned * BaseMesh::getIndices() const
{
	return _indices;
}
//:~
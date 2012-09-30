/*
 *  BaseMesh.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#ifdef WIN32
#include <windows.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
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

void BaseMesh::createVBOs()
{
	if(!_vertices) return;
	
	glGenBuffers(1, &_bufferedVertices);
	
	glBindBuffer(GL_ARRAY_BUFFER, _bufferedVertices);
	
	unsigned size = _numVertices * sizeof(Vector3F);
	glBufferData(GL_ARRAY_BUFFER, size, _vertices, GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
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

unsigned BaseMesh::getBufferedVertices() const
{
	return _bufferedVertices;
}
//:~
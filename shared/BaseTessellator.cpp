/*
 *  BaseTessellator.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/19/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseTessellator.h"

BaseTessellator::BaseTessellator() : m_cvs(0), m_indices(0), m_normals(0), m_uvs(0) 
{
	m_numVertices = m_numIndices = 0;
}

BaseTessellator::~BaseTessellator()
{
	cleanup();
}

void BaseTessellator::cleanup()
{
	if(m_cvs) delete[] m_cvs;
	if(m_indices) delete[] m_indices;
	if(m_normals) delete[] m_normals;
	if(m_uvs) delete[] m_uvs;
	m_cvs = 0;
	m_indices = 0;
	m_normals = 0;
	m_uvs = 0;
	m_numVertices = m_numIndices = 0;
}

void BaseTessellator::setNumVertices(unsigned x)
{
	m_numVertices = x;
}

void BaseTessellator::setNumIndices(unsigned x)
{
	m_numIndices = x;
}

void BaseTessellator::create(unsigned nv, unsigned ni)
{
	m_numIndices = ni;
	m_indices = new unsigned[m_numIndices];
	m_numVertices = nv;
	m_cvs = new Vector3F[m_numVertices];
	m_normals = new Vector3F[m_numVertices];
	m_uvs = new Vector2F[m_numVertices];
}

Vector3F * BaseTessellator::vertices()
{
	return m_cvs;
}

Vector3F * BaseTessellator::normals()
{
	return m_normals;
}

Vector2F * BaseTessellator::texcoords()
{
	return m_uvs;
}

unsigned * BaseTessellator::indices()
{
	return m_indices;
}

unsigned BaseTessellator::numVertices() const
{
	return m_numVertices;
}

unsigned BaseTessellator::numIndices() const
{
	return m_numIndices;
}
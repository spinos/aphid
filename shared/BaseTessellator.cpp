/*
 *  BaseTessellator.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/19/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseTessellator.h"

BaseTessellator::BaseTessellator() : m_cvs(0), m_indices(0), m_normals(0), m_uvs(0) {}

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
}

Vector3F * BaseTessellator::vertices() const
{
	return m_cvs;
}

Vector3F * BaseTessellator::normals() const
{
	return m_normals;
}

Vector2F * BaseTessellator::texcoords() const
{
	return m_uvs;
}

unsigned * BaseTessellator::indices() const
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
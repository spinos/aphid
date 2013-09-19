/*
 *  BaseTessellator.h
 *  mallard
 *
 *  Created by jian zhang on 9/19/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
class BaseTessellator {
public:
	BaseTessellator();
	virtual ~BaseTessellator();
	
	void cleanup();
	
	
	Vector3F * vertices() const;
	Vector3F * normals() const;
	unsigned * indices() const;
	unsigned numVertices() const;
	unsigned numIndices() const;
	
	Vector3F * m_cvs;
	Vector3F * m_normals;
	unsigned * m_indices;
	unsigned m_numVertices, m_numIndices;
private:

};
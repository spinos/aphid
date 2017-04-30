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
namespace aphid {

class BaseTessellator {
public:
	BaseTessellator();
	virtual ~BaseTessellator();
	
	virtual void cleanup();
	void create(unsigned nv, unsigned ni);
	
	void setNumVertices(unsigned x);
	void setNumIndices(unsigned x);

	Vector3F * vertices();
	Vector3F * normals();
	Vector2F * texcoords();
	unsigned * indices();
	unsigned numVertices() const;
	unsigned numIndices() const;
	
private:
	Vector3F * m_cvs;
	Vector3F * m_normals;
	Vector2F * m_uvs;
	unsigned * m_indices;
	unsigned m_numVertices, m_numIndices;
};

}
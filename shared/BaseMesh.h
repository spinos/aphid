/*
 *  BaseMesh.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Vector3F.h>

class BaseMesh {
public:
	BaseMesh();
	virtual ~BaseMesh();
	
	void createVertices(unsigned num);
	void createIndices(unsigned num);
	
	
	
	Vector3F * vertices();
	unsigned * indices();
	
	unsigned getNumVertices() const;
	unsigned getNumFaceVertices() const;
	Vector3F * getVertices() const;
	unsigned * getIndices() const;
	unsigned getBufferedVertices() const;
	
	Vector3F * _vertices;
	unsigned * _indices;
	unsigned _numVertices;
	unsigned _numFaceVertices;
	
};
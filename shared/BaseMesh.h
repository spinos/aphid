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
#include <Matrix33F.h>
#include <Geometry.h>
#include <Ray.h>
class BaseMesh : public Geometry {
public:
	BaseMesh();
	virtual ~BaseMesh();
	
	void verbose() const;
	
	void createVertices(unsigned num);
	void createIndices(unsigned num);
	const BoundingBox calculateBBox() const;
	const BoundingBox calculateBBox(const unsigned &idx) const;
	const int faceOnSideOf(const unsigned &idx, const int &axis, const float &pos) const;
	
	Vector3F * vertices();
	unsigned * indices();
	
	unsigned getNumFaces() const;
	unsigned getNumVertices() const;
	unsigned getNumFaceVertices() const;
	Vector3F * getVertices() const;
	unsigned * getIndices() const;
	virtual Matrix33F getTangentFrame(const unsigned& idx) const;
	
	char intersect(unsigned idx, const Ray & ray, Vector3F &hitP, Vector3F& hitN, float &hitD) const;
	
	Vector3F * _vertices;
	unsigned * _indices;
	unsigned _numVertices;
	unsigned _numFaces;
	unsigned _numFaceVertices;

};
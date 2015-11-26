/*
 *  tessellator.h
 *  catmullclark
 *
 *  Created by jian zhang on 11/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <Vector3F.h>
#include "bezierPatch.h"
//#include <zEXRImage.h>
class Tessellator {
public:
	Tessellator();
	~Tessellator();
	void cleanup();
	void setNumSeg(int n);
	//void setDisplacementMap(ZEXRImage* image);
	void evaluate(BezierPatch& bezier);
	void displacePositions(BezierPatch& bezier);
	void calculateNormals();
	void testLOD(BezierPatch& bezier);
	int numVertices() const;
	int numIndices() const;
	
	int* getVertices() const;
	float* getPositions() const;
	float* getNormals() const;
	Vector3F* p(int i, int j);
	Vector3F* n(int i, int j);
	
	int _numSeg;
	Vector3F* _positions;
	Vector3F* _normals;
	Vector3F* _texcoords;
	int* _vertices;
	//ZEXRImage* _displacementMap;
};
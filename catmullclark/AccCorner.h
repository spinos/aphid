/*
 *  AccCorner.h
 *  knitfabric
 *
 *  Created by jian zhang on 5/30/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Vector3F.h>

class AccCorner {
public:
	AccCorner() {}
	void reset();
	void addEdgeNeighbor(int idx, Vector3F * positions, Vector3F * normals);
	void addCornerNeighbor(int idx, Vector3F * positions, Vector3F * normals);
	void addCornerNeighborBetween(int a, int b, Vector3F * positions, Vector3F * normals);
	void edgeNeighborBeside(int nei, int & a, int &b) const;
	int valence() const;
	char isOnBoundary() const;
	Vector3F computeNormal() const;
	Vector3F computePosition() const;
	void verbose() const;
	Vector3F _edgePositions[5];
	Vector3F _cornerPositions[5];
	Vector3F _edgeNormals[5];
	Vector3F _cornerNormals[5];
	Vector3F _centerPosition, _centerNormal;
	int _edgeIndices[5];
	int _cornerIndices[5];
	int _numEdgeNei, _numCornerNei;
	int _centerIndex;
};
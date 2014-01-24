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
#include <vector>
class AccCorner {
public:
	AccCorner();
	virtual ~AccCorner();
	void setCenterIndex(const int & x);
	void setCenterPosition(Vector3F * p);
	void setCenterNormal(Vector3F * p);
	void reset();
	void addEdgeNeighbor(int idx, Vector3F * positions, Vector3F * normals);
	void addCornerNeighbor(int idx, Vector3F * positions, Vector3F * normals);
	void addCornerNeighborBetween(int a, int b, Vector3F * positions, Vector3F * normals);
	void edgeNeighborBeside(int nei, int & a, int &b) const;
	int valence() const;
	char isOnBoundary() const;
	Vector3F computeNormal() const;
	Vector3F computePosition() const;
	const int & edgeIndex(const unsigned & i) const;
	const std::vector<int> & edgeIndices() const;
	const std::vector<int> & cornerIndices() const;
	const std::vector<char> & tagCornerIndices() const;
	void verbose() const;
private:	
	Vector3F *_centerPosition, *_centerNormal;
	std::vector<Vector3F> _edgePositions;
	std::vector<Vector3F> _cornerPositions;
	std::vector<Vector3F> _edgeNormals;
	std::vector<Vector3F> _cornerNormals;
	std::vector<int> _edgeIndices;
	std::vector<int> _cornerIndices;
	std::vector<char> _tagCornerIndices;
	int _centerIndex;
};
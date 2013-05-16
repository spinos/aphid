/*
 *  accStencil.h
 *  catmullclark
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
class Vector3F;
class AccStencil {
public:
	AccStencil();
	~AccStencil();
	
	void setVertexPosition(Vector3F* data);
	void setVertexNormal(Vector3F* data);
	
	Vector3F computePositionOnCornerOnBoundary() const;
	Vector3F computePositionOnCorner() const;
	
	Vector3F computePositionOnEdgeOnBoundary() const;
	Vector3F computePositionOnEdge() const;
	
	Vector3F computePositionInterior() const;
	
	Vector3F computeNormalOnCornerOnBoundary() const;
	Vector3F computeNormalOnCorner() const;
	
	Vector3F computeNormalOnEdgeOnBoundary() const;
	Vector3F computeNormalOnEdge() const;
	
	Vector3F computeNormalInterior() const;

	Vector3F* _positions;
	Vector3F* _normals;
	int valence;
	int centerIndex;
	int edgeIndices[5];
	int cornerIndices[5];
};


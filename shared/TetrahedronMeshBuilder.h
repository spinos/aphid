/*
 *  TetrahedronMeshBuilder.h
 *  bcc
 *
 *  Created by jian zhang on 8/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <vector>
#include <GeometryArray.h>
#include <ATetrahedronMeshGroup.h>
#include <KdIntersection.h>
class TetrahedronMeshBuilder {
public:
	TetrahedronMeshBuilder();
	virtual ~TetrahedronMeshBuilder();
	
	virtual void build(GeometryArray * geos,
				unsigned & ntet, unsigned & nvert, unsigned & nstripes);
				
	virtual void addAnchors(ATetrahedronMesh * mesh, unsigned n, KdIntersection * anchorMesh);
	
	void getResult(ATetrahedronMeshGroup * m);
	
	static float EstimatedGroupSize;
protected:
	void addAnchor(ATetrahedronMesh * mesh, 
					unsigned istripe,
					const Vector3F & p, 
					unsigned tri);
	void addAnchor(ATetrahedronMesh * mesh, 
					KdIntersection * anchorMesh);
protected:
	std::vector<Vector3F > tetrahedronP;
	std::vector<unsigned > tetrahedronInd;
    std::vector<unsigned > pointDrifts;
    std::vector<unsigned > indexDrifts;
	
private:
};
/*
 *  SingleMeshBuilder.h
 *  bcc
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <TetrahedronMeshBuilder.h>
#include <AOrientedBox.h>
#include <deque>
class CartesianGrid;
class GeometryArray;
class SingleMeshBuilder : public TetrahedronMeshBuilder {
public:
	SingleMeshBuilder();
	virtual ~SingleMeshBuilder();
	
	virtual void build(GeometryArray * geos,
				unsigned & ntet, unsigned & nvert, unsigned & nstripes);
	 
	virtual void addAnchors(ATetrahedronMesh * mesh, unsigned n, KdIntersection * anchorMesh);
	
protected:
	void build(AOrientedBox * ob);
	
    void addAnchorBySide(ATetrahedronMesh * mesh, 
					unsigned istripe,
					const Matrix33F & invspace, 
					const Vector3F & center,
					const Vector3F toPoint,
                    float threashold,
					unsigned tri);
private:
	
private:
    AOrientedBox * m_boxes;
};

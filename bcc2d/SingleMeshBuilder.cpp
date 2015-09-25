/*
 *  SingleMeshBuilder.cpp
 *  bcc
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "SingleMeshBuilder.h"
#include <tetrahedron_math.h>
	
SingleMeshBuilder::SingleMeshBuilder() 
{
    m_boxes = NULL;
}
SingleMeshBuilder::~SingleMeshBuilder() 
{
    if(m_boxes) delete[] m_boxes;
}

void SingleMeshBuilder::build(GeometryArray * geos, 
					unsigned & ntet, unsigned & nvert, unsigned & nstripes)
{
	tetrahedronP.clear();
	tetrahedronInd.clear();
	pointDrifts.clear();
	indexDrifts.clear();
	
    const unsigned n = geos->numGeometries();
	if(m_boxes) delete[] m_boxes;
	m_boxes = new AOrientedBox[n];
	
    unsigned i=0;
    for(;i<n;i++) {
        pointDrifts.push_back(tetrahedronP.size());
        indexDrifts.push_back(tetrahedronInd.size());
		build((AOrientedBox *)geos->geometry(i));
		m_boxes[i] = *(AOrientedBox *)geos->geometry(i);
	}
	
	ntet = tetrahedronInd.size()/4;
	nvert = tetrahedronP.size();
	nstripes = n;
}

void SingleMeshBuilder::build(AOrientedBox * ob)
{
    const Vector3F center = ob->center();
    float w = ob->extent().x;
    float h = ob->extent().y * .67f;
    float d = ob->extent().y;
    
	Vector3F tetV[6];
	tetV[0] = center - Vector3F::ZAxis * h;
	tetV[1] = tetV[0] + Vector3F::ZAxis * h * 2.f;
	tetV[2] = center - Vector3F::XAxis * w
						- Vector3F::YAxis * d;
	tetV[3] = tetV[2] + Vector3F::YAxis * d * 2.f;
	tetV[4] = tetV[3] + Vector3F::XAxis * w * 2.f;
	tetV[5] = tetV[4] - Vector3F::YAxis * d * 2.f;
    
	Vector3F q;
    for(unsigned i = 0; i< 6; i++) {
		q = tetV[i];
		q -= ob->center();
		q = ob->orientation().transform(q);
		q += ob->center();
		tetV[i]  = q;
	}
	
    int i=0;
    for(;i<6;i++) tetrahedronP.push_back(tetV[i]);
    unsigned drift = pointDrifts.back();
    
    tetrahedronInd.push_back(drift + 0);
    tetrahedronInd.push_back(drift + 1);
    tetrahedronInd.push_back(drift + 2);
    tetrahedronInd.push_back(drift + 3);
    
    tetrahedronInd.push_back(drift + 0);
    tetrahedronInd.push_back(drift + 1);
    tetrahedronInd.push_back(drift + 3);
    tetrahedronInd.push_back(drift + 4);
    
    tetrahedronInd.push_back(drift + 0);
    tetrahedronInd.push_back(drift + 1);
    tetrahedronInd.push_back(drift + 4);
    tetrahedronInd.push_back(drift + 5);
    
    tetrahedronInd.push_back(drift + 0);
    tetrahedronInd.push_back(drift + 1);
    tetrahedronInd.push_back(drift + 5);
    tetrahedronInd.push_back(drift + 2);
}

void SingleMeshBuilder::addAnchors(ATetrahedronMesh * mesh, unsigned n, KdIntersection * anchorMesh)
{
	Vector3F endP[4];
	float lowerDist;
	unsigned anchorTri;
	int anchorSide;
	BoundingBox ab;
	Matrix33F invspace;
	unsigned i;
	Geometry::ClosestToPointTestResult cls;
	for(i=0;i<n;i++) {
		AOrientedBox & box = m_boxes[i];
		invspace = box.orientation();
		invspace.inverse();
// find which side of the box x-y plane is closer to grow mesh
		endP[0]  = box.majorPoint(true);
		cls.reset(endP[0], 1e8f);
		anchorMesh->closestToPoint(&cls);
		lowerDist = cls._distance;
		anchorTri = cls._icomponent;
        anchorSide = 0;
		
		endP[1] = box.minorPoint(false);
		cls.reset(endP[1], 1e8f);
		anchorMesh->closestToPoint(&cls);
		
		if(cls._distance < lowerDist) {
			anchorTri = cls._icomponent;
            anchorSide = 1;
		}
		//addAnchorByThreshold(mesh, i, invspace, box.center(), 
		//					anchorX, isLowerEnd, anchorTri);
	}
}
/*
void SingleMeshBuilder::addAnchorByThreshold(ATetrahedronMesh * mesh, 
					unsigned istripe,
					const Matrix33F & invspace, 
					const Vector3F & center,
					float threshold,
					bool isLower,
					unsigned tri)
{
	unsigned tetMax = mesh->numTetrahedrons() * 4;
	if(istripe < indexDrifts.size() -1)
		tetMax = indexDrifts[istripe + 1];
		
	BoundingBox box;
	Vector3F q;
	unsigned i = indexDrifts[istripe];
	unsigned j;
	for(;i<tetMax;i+=4) {
		unsigned * tet = mesh->tetrahedronIndices(i/4);
		box.reset();
        for(j=0; j< 4; j++)
            box.expandBy(mesh->points()[tet[j]], 1e-3f); 
			
		q = invspace.transform(box.center() - center);
		if(isLower) {
			if(q.x > threshold) continue;
		}
		else {
			if(q.x < threshold) continue;
		}
		
		for(j=0; j< 4; j++)
			mesh->anchors()[tet[j]] = (1<<24 | tri);
	}
}*/
//:~

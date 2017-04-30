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
	unsigned i, j;
	Geometry::ClosestToPointTestResult cls;
	for(i=0;i<n;i++) {
		AOrientedBox & box = m_boxes[i];
		invspace = box.orientation();
		invspace.inverse();
// find which side of the box x-y plane is closer to grow mesh
		endP[0]  = box.majorPoint(true);
        endP[1] = box.minorPoint(true);
        endP[2] = box.majorPoint(false);
        endP[3] = box.minorPoint(false);
        
        lowerDist = 1e8f;
        anchorSide = 0;
        for(j = 0; j<4; j++) {
            cls.reset(endP[j], 1e8f);
            anchorMesh->closestToPoint(&cls);
            
            if(lowerDist > cls._distance) {
                lowerDist = cls._distance;
                anchorTri = cls._icomponent;
                anchorSide = j;
            }   
        }
        
        float threshold = box.extent().y * 1.01f;
        if(anchorSide == 1 || anchorSide == 3) threshold = box.extent().x * 1.01f;
		addAnchorBySide(mesh, i, invspace, box.center(), 
						endP[anchorSide], 
                        threshold,
                        anchorTri);
	}
}

void SingleMeshBuilder::addAnchorBySide(ATetrahedronMesh * mesh, 
					unsigned istripe,
					const Matrix33F & invspace, 
					const Vector3F & center,
					const Vector3F toPoint,
                    float threashold,
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
		for(j=0; j<4; j++) {
            if( toPoint.distanceTo( mesh->points()[tet[j]] ) < threashold )
                mesh->anchors()[tet[j]] = (1<<24 | tri);
        }
	}
}
//:~

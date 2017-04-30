/*
 *  TriangleAnchorDeformer.cpp
 *  aphid
 *
 *  Created by jian zhang on 7/19/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "TriangleAnchorDeformer.h"
#include "TriangleDifference.h"
#include "AGenericMesh.h"
#include "BaseBuffer.h"
#include "ATriangleMesh.h"
TriangleAnchorDeformer::TriangleAnchorDeformer() 
{
	m_localP = new BaseBuffer;
}

TriangleAnchorDeformer::~TriangleAnchorDeformer() 
{
	delete m_localP;
}

void TriangleAnchorDeformer::setDifference(TriangleDifference * diff)
{ m_diff = diff; }

void TriangleAnchorDeformer::setMesh(AGenericMesh * mesh)
{
	m_localP->create(mesh->numPoints()*12);
	ADeformer::setMesh(mesh);
}

void TriangleAnchorDeformer::reset(ATriangleMesh * restM)
{
	const unsigned n = numVertices();
	unsigned * a = mesh()->anchors();
	Vector3F * src = restP();
	Vector3F * dst = localP();
	unsigned i=0;
	for(;i<n;i++) {
		if(a[i]>0) {
			const unsigned ia = (a[i]<<8)>>8;
			dst[i] = src[i] - restM->triangleCenter(ia);
		}
	}
	ADeformer::reset();
}

Vector3F * TriangleAnchorDeformer::localP() const
{ return (Vector3F *)m_localP->data(); }

bool TriangleAnchorDeformer::solve(ATriangleMesh * m)
{
	Matrix44F sp;
	const unsigned n = numVertices();
	unsigned * a = mesh()->anchors();
	Vector3F * dst = deformedP();
	Vector3F * src = localP();
	unsigned i=0;
	for(;i<n;i++) {
		if(a[i]>0) {
			const unsigned ia = (a[i]<<8)>>8;
			Matrix33F q = m_diff->Q()[ia];
            q.orthoNormalize();
			const Vector3F t = m->triangleCenter(ia);
			sp.setRotation(q);
			sp.setTranslation(t);
			dst[i] = sp.transform(src[i]);
		}
	}
	return true;
}
/*
 *  FeatherDeformer.cpp
 *  
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherDeformer.h"
#include "FeatherMesh.h"
#include <math/Matrix44F.h>
#include <ConvexShape.h>

using namespace aphid;

FeatherDeformer::FeatherDeformer(const FeatherMesh * mesh)
{ 
	m_mesh = mesh; 
	m_points.reset(new Vector3F[mesh->numPoints() ]);
	m_normals.reset(new Vector3F[mesh->numPoints() ]);
	m_leadingEdgePoints.reset(new Vector3F[mesh->numLeadingEdgeVertices() ]);
}

FeatherDeformer::~FeatherDeformer()
{}

const Vector3F * FeatherDeformer::deformedPoints() const
{ return m_points.get(); }

const Vector3F * FeatherDeformer::deformedNormals() const
{ return m_normals.get(); }

const Vector3F * FeatherDeformer::deformedLeadingEdgePoints() const
{ return m_leadingEdgePoints.get(); }

void FeatherDeformer::deform(const Matrix33F & mat)
{
	memcpy( m_points.get(), m_mesh->points(), m_mesh->numPoints() * 12 );
	
	Matrix44F matStep;
	matStep.setRotation(mat);
	
	Matrix44F acct;
	
	float xMean;
	float lastXMean = 0.f;
	const int & nvpr = m_mesh->numVerticesPerRow();
	const int & nv = m_mesh->numPoints();
	int it = nv - 1 - m_mesh->vertexFirstRow();
	for(;it >= 0;it -= nvpr) {
		
		int rowEnd = it - nvpr;
		if(rowEnd < -1) {
			rowEnd = -1;
		}
		
/// mean of current step
		xMean = 0.f;
		for(int i=it;i>rowEnd;--i) {
			xMean += m_points[i].x;
		}
		xMean /= (float)(it - rowEnd);
		
/// relative to last step
		for(int i=it;i>rowEnd;--i) {
			m_points[i].x -= lastXMean;
		}
		
/// local warp
		for(int i=it;i>rowEnd;--i) {
			m_points[i] = acct.transform(m_points[i]);
		}

/// accumulate each step
		matStep.setTranslation(xMean - lastXMean,0,0);
		acct = matStep * acct;
		lastXMean = xMean;
		
	}

/// update leading edge
	const int & nev = m_mesh->numLeadingEdgeVertices();
	const int * eind = m_mesh->leadingEdgeIndices();
	for(int i=0;i<nev;++i) {
		m_leadingEdgePoints[i] = m_points[eind[i]];
	}
}

void FeatherDeformer::calculateNormal()
{
	const int nv = m_mesh->numPoints();
	std::memset(m_normals.get(), 0, nv * 12);
	const int ni = m_mesh->numIndices();
	const unsigned * ind = m_mesh->indices();
	const Vector3F * ps = m_points.get();
	
	cvx::Triangle atri;
	Vector3F triNm;

	for(int i=0;i<ni;i+=3) {
		const unsigned & i0 = ind[i];
		const unsigned & i1 = ind[i+1];
		const unsigned & i2 = ind[i+2];
		
		const Vector3F & a = ps[i0];
		const Vector3F & b = ps[i1];
		const Vector3F & c = ps[i2];
		
		atri.set(a, b, c);
		triNm = atri.calculateNormal();
		
		m_normals[i0] += triNm;
		m_normals[i1] += triNm;
		m_normals[i2] += triNm;
	}
	
	for(int i=0;i<nv;++i) {
		m_normals[i].normalize();
	}

}

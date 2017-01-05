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

using namespace aphid;

FeatherDeformer::FeatherDeformer(const FeatherMesh * mesh)
{ 
	m_mesh = mesh; 
	m_points.reset(new Vector3F[mesh->numPoints() ]);
	
}

FeatherDeformer::~FeatherDeformer()
{}

const Vector3F * FeatherDeformer::deformedPoints() const
{ return m_points.get(); }

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
	
}
/*
 *  BendTwistRollDeformer.cpp
 *  
 *  bend effect > twist effect > roll
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "BendTwistRollDeformer.h"
#include "geom/ATriangleMesh.h"
#include <math/Matrix44F.h>
#include <geom/ConvexShape.h>

using namespace aphid;

BendTwistRollDeformer::BendTwistRollDeformer(const ATriangleMesh * mesh)
{ 
	m_mesh = mesh; 
	m_points.reset(new Vector3F[mesh->numPoints() ]);
	m_normals.reset(new Vector3F[mesh->numPoints() ]);
	m_warpAngle[0] = m_warpAngle[1] = 0.f;
}

BendTwistRollDeformer::~BendTwistRollDeformer()
{}

const Vector3F * BendTwistRollDeformer::deformedPoints() const
{ return m_points.get(); }

const Vector3F * BendTwistRollDeformer::deformedNormals() const
{ return m_normals.get(); }

void BendTwistRollDeformer::deform(const Matrix33F & mat)
{
	memcpy( m_points.get(), m_mesh->points(), m_mesh->numPoints() * 12 );
	
	Matrix44F matStep;
	matStep.setRotation(mat);
	
	Matrix44F acct;
	
	float xMean;
	float lastXMean = 0.f;
	const int & nvpr = m_mesh->numVerticesPerRow();
	const int & nv = m_mesh->numPoints();
	const int & nrow = (nv-2) / nvpr;
	const float drow = 1.f / (float)nrow;
	
	int it = nv - 1 - m_mesh->vertexFirstRow();
	int irow = 0;
	for(;it >= 0;it -= nvpr, irow++) {
		
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
		Quaternion q(m_warpAngle[0] + m_warpAngle[1] * (drow * irow), Vector3F::XAxis);
		Matrix33F mwarp(q);
		for(int i=it;i>rowEnd;--i) {
			
			m_points[i] = mwarp.transform(m_points[i]);
		}

		for(int i=it;i>rowEnd;--i) {
			m_points[i] = acct.transform(m_points[i]);
		}

/// accumulate each step
		matStep.setTranslation(xMean - lastXMean,0,0);
		acct = matStep * acct;
		lastXMean = xMean;
		
	}
	
}

void BendTwistRollDeformer::calculateNormal()
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

void BendTwistRollDeformer::setWarpAngles(const float * v)
{ 
	m_warpAngle[0] = v[1];
	m_warpAngle[1] = v[0] - v[1];
}

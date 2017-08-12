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

namespace aphid {

BendTwistRollDeformer::BendTwistRollDeformer()
{ 
    m_np = 0;
    memset(m_angles, 0, 12);
}

BendTwistRollDeformer::~BendTwistRollDeformer()
{}

void BendTwistRollDeformer::setBend(const float& x)
{ m_angles[0] = x; }

void BendTwistRollDeformer::setTwist(const float& x)
{ m_angles[1] = x; }

void BendTwistRollDeformer::setRoll(const float& x)
{ m_angles[2] = x; }

void BendTwistRollDeformer::setOriginalMesh(const ATriangleMesh * mesh)
{
	if(!mesh) 
	    return;
	
	int np = mesh->numPoints();
	if(m_np < np) {
	    m_points.reset(new Vector3F[np]);
	    m_normals.reset(new Vector3F[np]);
	}
	
	memcpy( m_points.get(), mesh->points(), np * 12 );
	m_np = np;
}

const Vector3F * BendTwistRollDeformer::deformedPoints() const
{ return m_points.get(); }

const Vector3F * BendTwistRollDeformer::deformedNormals() const
{ return m_normals.get(); }

void BendTwistRollDeformer::deform(const ATriangleMesh * mesh)
{
    if(!mesh)
        return;
    
    setOriginalMesh(mesh);
    
	const int & nv = mesh->numPoints();
	
	float yMean;
	int rownv;
/// first row
	getRowMean(0, nv, rownv, yMean);
	float lastYMean = yMean;
	
/// scaled by num_rows
	float deltaBend = (float)rownv / (float)nv * m_angles[0] * 3.3f;
	
	Matrix33F mat;
/// bend backward for positive values
    mat.rotateX(-deltaBend);
    mat.rotateZ(m_angles[2]);
	Matrix44F matStep;
	matStep.setRotation(mat);
	
	Matrix44F acct;
	
	int irow = 1;
	for(int it = rownv;it < nv;it+=rownv) {
				
/// mean of current step
		getRowMean(it, nv, rownv, yMean);
		
/// relative to last step
		for(int i=0;i<rownv;++i) {
			m_points[it + i].y -= lastYMean;
		}
		
/// local warp
		Quaternion q(m_angles[1] * irow, Vector3F::YAxis);
		Matrix33F mwarp(q);

		for(int i=0;i<rownv;++i) {			
			m_points[it + i] = mwarp.transform(m_points[it + i]);
		}

		for(int i=0;i<rownv;++i) {
			m_points[it + i] = acct.transform(m_points[it + i]);
		}
		
/// accumulate each step
		matStep.setTranslation(0, yMean - lastYMean, 0);
		acct = matStep * acct;
		lastYMean = yMean;
		irow++;
	}
	
	calculateNormal(mesh);
	
}

float BendTwistRollDeformer::getRowMean(int rowBegin, int nv, int& nvRow, float& rowBase ) const
{
	rowBase = m_points[rowBegin].y;
	
	float res = 0.f;
	nvRow = 0;
	for(int i=rowBegin;i<nv;++i) {
		if(m_points[i].y > rowBase + 1e-3f)
			break;
			
		res += m_points[i].y;
		nvRow++;
	}
	res /= (float)nvRow;
	return res;
}

void BendTwistRollDeformer::calculateNormal(const ATriangleMesh * mesh)
{
	const int nv = mesh->numPoints();
	std::memset(m_normals.get(), 0, nv * 12);
	const int ni = mesh->numIndices();
	const unsigned * ind = mesh->indices();
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

}

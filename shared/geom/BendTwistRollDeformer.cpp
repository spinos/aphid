/*
 *  BendTwistRollDeformer.cpp
 *  
 *  deform a billboard by bend x twist y roll z
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

SplineMap1D* BendTwistRollDeformer::weightSpline()
{ return &m_weightSpline; }

void BendTwistRollDeformer::computeRowWeight(const ATriangleMesh * mesh)
{
	const int nRows = GetNumRows(mesh); 
	
	m_weights.reset(new float[nRows]);
	
	const float drow = 1.f/(float)(nRows - 1);
	float weightSum = 0.f;
	for(int i=0;i<nRows;++i) {
		float d = m_weightSpline.interpolate(drow * i);
		if(d < 0.04f)
			d = 0.04f;
		m_weights[i] = d;
		weightSum += d;
	}
	
	weightSum = 1.f / weightSum;
	for(int i=0;i<nRows;++i) {
		m_weights[i] *= weightSum;
	}
}

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
	
	
	Matrix44F matStep;
	
	Matrix44F acct;
	Matrix33F twistmat;
	
	int irow = 1;
	for(int it = rownv;it < nv;it+=rownv) {
				
/// mean of current step
		getRowMean(it, nv, rownv, yMean);
		
/// relative to last step
		for(int i=0;i<rownv;++i) {
			points()[it + i].y -= lastYMean;
		}
		
		const float& wei = m_weights[irow];
		
/// local warp
		Quaternion q(m_angles[1] * wei * 4.f, Vector3F::YAxis);
		Matrix33F mwarp(q);
		twistmat *= mwarp;

		for(int i=0;i<rownv;++i) {			
			points()[it + i] = twistmat.transform(points()[it + i]);
		}

		for(int i=0;i<rownv;++i) {
			points()[it + i] = acct.transform(points()[it + i]);
		}
		
/// accumulate each step
		Matrix33F rotmat;
/// bend backward for positive values
		rotmat.rotateX(-m_angles[0] * wei * 3.9f);
		rotmat.rotateZ(m_angles[2] * wei * 2.9f);
		
		matStep.setRotation(rotmat);
		matStep.setTranslation(0, yMean - lastYMean, 0);
		acct = matStep * acct;
		lastYMean = yMean;
		irow++;
	}
	
	calculateNormal(mesh);
	
}

const float& BendTwistRollDeformer::rowWeight(int i) const
{ return m_weights[i]; }

const float& BendTwistRollDeformer::bendAngle() const
{ return m_angles[0]; }

const float& BendTwistRollDeformer::twistAngle() const
{ return m_angles[1]; }

const float& BendTwistRollDeformer::rollAngle() const
{ return m_angles[2]; }

}

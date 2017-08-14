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
	float deltaRoll = (float)rownv / (float)nv * m_angles[2] * 2.9f;
	
	Matrix33F mat;
/// bend backward for positive values
    mat.rotateX(-deltaBend);
    mat.rotateZ(deltaRoll);
	Matrix44F matStep;
	matStep.setRotation(mat);
	
	Matrix44F acct;
	
	int irow = 1;
	for(int it = rownv;it < nv;it+=rownv) {
				
/// mean of current step
		getRowMean(it, nv, rownv, yMean);
		
/// relative to last step
		for(int i=0;i<rownv;++i) {
			points()[it + i].y -= lastYMean;
		}
		
/// local warp
		Quaternion q(m_angles[1] * irow, Vector3F::YAxis);
		Matrix33F mwarp(q);

		for(int i=0;i<rownv;++i) {			
			points()[it + i] = mwarp.transform(points()[it + i]);
		}

		for(int i=0;i<rownv;++i) {
			points()[it + i] = acct.transform(points()[it + i]);
		}
		
/// accumulate each step
		matStep.setTranslation(0, yMean - lastYMean, 0);
		acct = matStep * acct;
		lastYMean = yMean;
		irow++;
	}
	
	calculateNormal(mesh);
	
}

}

/*
 *  FoldCrumpleDeformer.cpp
 *  
 *  bend x twist y roll z and folding
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FoldCrumpleDeformer.h"
#include "geom/ATriangleMesh.h"
#include <math/Matrix44F.h>
#include <geom/ConvexShape.h>

namespace aphid {

FoldCrumpleDeformer::FoldCrumpleDeformer()
{}

FoldCrumpleDeformer::~FoldCrumpleDeformer()
{}

void FoldCrumpleDeformer::deform(const ATriangleMesh * mesh)
{
    if(!mesh)
        return;
    
    setOriginalMesh(mesh);
    
	const int & nv = mesh->numPoints();
	
	float yMean;
	int rownv;
/// first row
	getRowMean(0, nv, rownv, yMean);
	foldARow(0, rownv, 0.f);
	
	float lastYMean = yMean;
	
	Matrix44F matStep;
	Matrix44F acct;
	Matrix33F twistmat;
	
	float crumplePhase = 0.f;
	
	const float drow = 1.f /(float)(nv/rownv);
	int irow = 1;
	for(int it = rownv;it < nv;it+=rownv) {
				
/// mean of current step
		getRowMean(it, nv, rownv, yMean);
		
/// relative to last step
		for(int i=0;i<rownv;++i) {
			points()[it + i].y -= lastYMean;
		}
		
		foldARow(it, rownv, drow * irow);
		
		const float& wei = rowWeight(irow);		

/// local warp
		Quaternion q(twistAngle() * wei * 4.f, Vector3F::YAxis);
		Matrix33F mwarp(q);
		twistmat *= mwarp;
		
		float Kcrumple = m_crumpleSpline.interpolate(drow * irow);
		Quaternion qcrumple(Kcrumple * sin(crumplePhase) * m_crumpleAngle, Vector3F::YAxis);
		Matrix33F crumplemat(qcrumple);
		crumplemat *= twistmat;
		
		for(int i=0;i<rownv;++i) {			
			points()[it + i] = crumplemat.transform(points()[it + i]);
		}

		for(int i=0;i<rownv;++i) {
			points()[it + i] = acct.transform(points()[it + i]);
		}
		
/// accumulate each step
		Matrix33F rotmat;
/// bend backward for positive values
		rotmat.rotateX(-bendAngle() * wei * 3.9f);
		rotmat.rotateZ(rollAngle() * wei * 2.9f);
		
		matStep.setRotation(rotmat);
		matStep.setTranslation(0, yMean - lastYMean, 0);
		acct = matStep * acct;
		lastYMean = yMean;
		irow++;
		crumplePhase += drow * 12.f + RandomF01();
	}
	
	calculateNormal(mesh);
	
}

void FoldCrumpleDeformer::foldARow(const int& rowBegin, const int& rownv,
					const float& rowparam)
{
	const int centeri = rownv>>1;
	float Kfold = m_foldSpline.interpolate(rowparam);
	Quaternion qfold(Kfold * m_foldAngle, Vector3F::YAxis);
	Matrix33F foldmat(qfold);
			
	for(int i=0;i<rownv;++i) {
		if(i == centeri) {
			qfold = Quaternion(-Kfold * m_foldAngle, Vector3F::YAxis);
			foldmat = Matrix33F(qfold);
		} else {
			points()[rowBegin + i] = foldmat.transform(points()[rowBegin + i]);
		}
	}
}

SplineMap1D* FoldCrumpleDeformer::foldSpline()
{ return &m_foldSpline; }

SplineMap1D* FoldCrumpleDeformer::crumpleSpline()
{ return &m_crumpleSpline; }

void FoldCrumpleDeformer::setCrumple(const float& x)
{ m_crumpleAngle = x; }

void FoldCrumpleDeformer::setFold(const float& x)
{ m_foldAngle = x; }

}

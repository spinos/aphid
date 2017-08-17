/*
 *  DirectionalBendDeformer.cpp
 *  
 *  bend effect > twist effect > roll
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DirectionalBendDeformer.h"
#include "geom/ATriangleMesh.h"
#include <math/Matrix44F.h>
#include <geom/ConvexShape.h>

namespace aphid {

DirectionalBendDeformer::DirectionalBendDeformer()
{
	m_tip = new Vector3F(0,1,0);
}

DirectionalBendDeformer::~DirectionalBendDeformer()
{ delete m_tip; }

void DirectionalBendDeformer::setDirection(const Vector3F& v)
{ 
	*m_tip = v.normal(); 
	m_orientation = Vector3F(v.x, 0.f, v.z).orientation();
}

void DirectionalBendDeformer::deform(const ATriangleMesh * mesh)
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
	
	Matrix44F acct;
	getBaseMat(acct);
/// first row
	for(int i = 0;i < rownv;++i) {
		points()[i] = acct.transform(points()[i]);
	}
		
	int irow = 1;
	for(int it = rownv;it < nv;it+=rownv) {
				
/// mean of current step
		getRowMean(it, nv, rownv, yMean);
		
/// relative to last step
		for(int i=0;i<rownv;++i) {
			points()[it + i].y -= lastYMean;
		}

		for(int i=0;i<rownv;++i) {
			points()[it + i] = acct.transform(points()[it + i]);
		}
		
		const float& wei = m_bendWeight[irow];

/// locat rotation						
		Matrix44F matStep;
		getRotMat(matStep, wei, m_noiseWeight[irow] * RandomFn11() );
		
/// local translate		
		matStep.setTranslation(Vector3F(0, yMean - lastYMean, 0) );
/// accumulate each step		
		acct = matStep * acct;
		lastYMean = yMean;
		irow++;
	}
	
	calculateNormal(mesh);
	
}

void DirectionalBendDeformer::getBaseMat(Matrix44F& mat)
{
	const float noi = .1f * RandomFn11();
	Matrix33F thetamat;
	switch (m_orientation) {
		case 0:
			thetamat.rotateY(-atan(m_tip->z / m_tip->x) + noi );
		break;
		case 1:
			thetamat.rotateY(-atan(m_tip->z / m_tip->x) + noi );
		break;
		case 4:
			thetamat.rotateY(atan(m_tip->x / m_tip->z) + noi );
		break;
		default:
			thetamat.rotateY(atan(m_tip->x / m_tip->z) + noi );
			break;
	}
	mat.setRotation(thetamat);

}

void DirectionalBendDeformer::getRotMat(Matrix44F& mat, const float& wei,
					const float& noi)
{
/// angle to zenith
	float phi = acos(m_tip->y);
	
	Matrix33F phimat;
	switch (m_orientation) {
		case 0:
			phimat.rotateZ(phi* (wei + noi));
		break;
		case 1:
			phimat.rotateZ(-phi* (wei + noi));
		break;
		case 4:
			phimat.rotateX(-phi* (wei + noi));
		break;
		default:
			phimat.rotateX(phi* (wei + noi));
			break;
	}
	Matrix33F thetamat;
	thetamat.rotateY( noi * 1.5f );
	mat.setRotation(phimat * thetamat);
	
}

void DirectionalBendDeformer::computeRowWeight(const ATriangleMesh * mesh)
{
	const int nRows = GetNumRows(mesh); 
	
	m_bendWeight.reset(new float[nRows]);
	m_noiseWeight.reset(new float[nRows]);
	
	const float drow = 1.f/(float)(nRows - 1);
	float bendSum = 0.f;
	for(int i=0;i<nRows;++i) {
		float d = m_bendSpline.interpolate(drow * i);
		bendSum += d;
		m_bendWeight[i] = d;
		
		d = m_noiseSpline.interpolate(drow * i);
		m_noiseWeight[i] = d * .25f;
	}
	
	bendSum = 1.f / bendSum;
	for(int i=0;i<nRows;++i) {
		m_bendWeight[i] *= bendSum;
	}
}

SplineMap1D* DirectionalBendDeformer::bendSpline()
{ return &m_bendSpline; }

SplineMap1D* DirectionalBendDeformer::noiseSpline()
{ return &m_noiseSpline; }


}

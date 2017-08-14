/*
 *  DirectionalBendDeformer.h
 *  
 *  deform a billboard
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef BEND_TWIST_ROLL_DEFORMER_H
#define BEND_TWIST_ROLL_DEFORMER_H

#include "TriangleMeshDeformer.h"
#include <math/SplineMap1D.h>

namespace aphid {

class Matrix44F;

class DirectionalBendDeformer : public TriangleMeshDeformer {

	boost::scoped_array<float> m_bendWeight;
	boost::scoped_array<float> m_noiseWeight;
	SplineMap1D m_bendSpline;
	SplineMap1D m_noiseSpline;
	Vector3F* m_tip;
	int m_orientation;
	
public:
    DirectionalBendDeformer();
	virtual ~DirectionalBendDeformer();
	
	void setDirection(const Vector3F& v);
	
	virtual void deform(const ATriangleMesh * mesh);
	
	void computeRowWeight(const ATriangleMesh * mesh);
	
	SplineMap1D* bendSpline();
	SplineMap1D* noiseSpline();
	
protected:
	
private:
	void getBaseMat(Matrix44F& mat);
	void getRotMat(Matrix44F& mat, const float& wei,
					const float& noi);
	
};

}
#endif

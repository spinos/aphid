/*
 *  FoldCrumpleDeformer.h
 *  
 *  bend x twist y roll z and folding
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FOLD_CRUMPLE_DEFORMER_H
#define FOLD_CRUMPLE_DEFORMER_H

#include "BendTwistRollDeformer.h"

namespace aphid {

class FoldCrumpleDeformer : public BendTwistRollDeformer {

	SplineMap1D m_foldSpline;
	SplineMap1D m_crumpleSpline;
	float m_crumpleAngle;
	float m_foldAngle;
	
public:
    FoldCrumpleDeformer();
	virtual ~FoldCrumpleDeformer();
	
	virtual void deform(const ATriangleMesh * mesh);
	
	SplineMap1D* foldSpline();
	SplineMap1D* crumpleSpline();
	void setCrumple(const float& x);
	void setFold(const float& x);
	
protected:
	void foldARow(const int& rowBegin, const int& rownv,
					const float& rowparam);
	
private:
	
};

}
#endif

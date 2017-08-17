/*
 *  BendTwistRollDeformer.h
 *  
 *  deform a billboard by bend x twist y roll z
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

class BendTwistRollDeformer : public TriangleMeshDeformer {

	boost::scoped_array<float> m_weights;
	SplineMap1D m_weightSpline;
/// bend-x, twist-y, roll-z rotation
	float m_angles[3];
	
public:
    BendTwistRollDeformer();
	virtual ~BendTwistRollDeformer();
	
	void setBend(const float& x);
	void setTwist(const float& x);
	void setRoll(const float& x);
	void computeRowWeight(const ATriangleMesh * mesh);
	
	virtual void deform(const ATriangleMesh * mesh);
	
	SplineMap1D* weightSpline();
	
protected:
	const float& rowWeight(int i) const;
	const float& bendAngle() const;
	const float& twistAngle() const;
	const float& rollAngle() const;
	
private:
	
};

}
#endif

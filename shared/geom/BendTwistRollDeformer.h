/*
 *  BendTwistRollDeformer.h
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

namespace aphid {

class BendTwistRollDeformer : public TriangleMeshDeformer {

/// bend-x, twist-y, roll-z rotation
	float m_angles[3];
	
public:
    BendTwistRollDeformer();
	virtual ~BendTwistRollDeformer();
	
	void setBend(const float& x);
	void setTwist(const float& x);
	void setRoll(const float& x);
	
	virtual void deform(const ATriangleMesh * mesh);
	
protected:
	
private:
	
};

}
#endif

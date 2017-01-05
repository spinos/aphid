/*
 *  FeatherDeformParam.h
 *  cinchona
 *
 *  Created by jian zhang on 1/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FEATHER_DEFORM_PARAM_H
#define FEATHER_DEFORM_PARAM_H

#include "FeatherOrientationParam.h"

class FeatherDeformParam : public FeatherOrientationParam {

public:
	FeatherDeformParam();
	virtual ~FeatherDeformParam();
	
	void predictRotation(aphid::Matrix33F & dst,
						const float * x,
						const float & relspeed);
	
protected:
	
private:
	
};
#endif

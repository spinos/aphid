/*
 *  QuadraticCurve.h
 *  knitfabric
 *
 *  Created by jian zhang on 5/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <BaseCurve.h>

class QuadraticCurve : public BaseCurve {
public:
	QuadraticCurve();
	virtual ~QuadraticCurve();
	
	virtual Vector3F interpolate(float param) const;
private:
};

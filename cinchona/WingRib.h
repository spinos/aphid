/*
 *  WingRib.h
 *  cinchona
 *
 *  Created by jian zhang on 1/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef WING_RIB_H
#define WING_RIB_H

#include <geom/Airfoil.h>
#include <math/Matrix44F.h>

class WingRib : public aphid::Airfoil, public aphid::Matrix44F {

public:
	WingRib(const float & c,
			const float & m,
			const float & p,
			const float & t);
	virtual ~WingRib();
	
/// point on outline
/// [  0,  1.0  ] upper
/// [-1.0,-0.001] lower 
	void getPoint(aphid::Vector3F & dst, const float & param) const;
	
private:
};

#endif
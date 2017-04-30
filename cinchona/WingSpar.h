/*
 *  WingSpar.h
 *  cinchona
 *
 *  Created by jian zhang on 1/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef WING_SPAR_H
#define WING_SPAR_H

#include <math/HermiteInterpolatePiecewise.h>

namespace aphid {

class Vector3F;

}

class WingSpar : public aphid::HermiteInterpolatePiecewise<float, aphid::Vector3F > {

public:
	WingSpar(const int & np);
	virtual ~WingSpar();
	
/// i 0:np
	void setKnot(const int & i,
				const aphid::Vector3F & p,
				const aphid::Vector3F & t);
	
/// i 0:99
	void getPoint(aphid::Vector3F & dst, const int & i) const;
	
	aphid::Vector3F getPoint(const int & idx,
				const float & param) const;
				
protected:

private:

};

#endif
/*
 *  HemisphereMesh.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseMesh.h>

class HemisphereMesh : public BaseMesh {
public:
	HemisphereMesh(unsigned grid_theta, unsigned grid_phi);
	virtual ~HemisphereMesh();
	
	unsigned getGridTheta() const;
	unsigned getGridPhi() const;
private:
	unsigned _gridTheta, _gridPhi;
};
/*
 *  BRDFProgram.h
 *  
 *
 *  Created by jian zhang on 10/2/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <HemisphereProgram.h>

class BRDFProgram : public HemisphereProgram {
public:
	BRDFProgram();
	
	static void setVTheta(float val);
	static Vector3F V, N, Tangent, Binormal;
};

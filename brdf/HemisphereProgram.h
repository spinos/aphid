/*
 *  HemisphereProgram.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <CUDAProgram.h>
#include <HemisphereMesh.h>

class HemisphereProgram : public CUDAProgram {
public:
	HemisphereProgram();
	virtual ~HemisphereProgram();
	
	virtual void run(CUDABuffer * buffer, BaseMesh * mesh);
};
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

class BezierProgram : public CUDAProgram {
public:
	BezierProgram();
	virtual ~BezierProgram();
	
	virtual void run(CUDABuffer * buffer, BaseMesh * mesh);
};
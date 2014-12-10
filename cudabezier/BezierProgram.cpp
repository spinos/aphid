/*
 *  HemisphereProgram.cpp
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BezierProgram.h"
#include "bezier_implement.h"

BezierProgram::BezierProgram() {}
BezierProgram::~BezierProgram() {}

void BezierProgram::run(CUDABuffer * buffer, CUDABuffer * cvs, BaseCurve * curve)
{
	void *dptr;
	buffer->map(&dptr);
	cvs->hostToDevice(curve->m_cvs, curve->numVertices() * 12);
	
	hemisphere((float4 *)dptr, (float3 *)cvs->bufferOnDevice(), curve->numVertices());

	buffer->unmap();
}
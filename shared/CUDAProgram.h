/*
 *  CUDAProgram.h
 *  
 *
 *  Created by jian zhang on 10/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <CUDABuffer.h>
class CUDAProgram {
public:
	CUDAProgram();
	virtual ~CUDAProgram();
	
	virtual void run(CUDABuffer * buffer);
	
	void map(CUDABuffer * buffer, void ** p);
	void unmap(CUDABuffer * buffer);

};
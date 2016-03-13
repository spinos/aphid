/*
 *  CubeRender.h
 *  
 *
 *  Created by jian zhang on 3/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <CudaRender.h>

namespace aphid {

class CubeRender : public CudaRender {

public:
	CubeRender();
	virtual ~CubeRender();
	
	virtual void setBufferSize(const int & w, const int & h);
	virtual void render();
	
};

}
/*
 *  CubeRender.h
 *  
 *
 *  Created by jian zhang on 3/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <CudaRender.h>

class CubeRender : public aphid::CudaRender {

public:
	CubeRender();
	virtual ~CubeRender();
	
	virtual void setSize(const int & w, const int & h);
	void render();
	
};
/*
 *  WorldRender.h
 *  julia
 *
 *  Created by jian zhang on 3/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <CudaRender.h>

namespace aphid {

class WorldRender : public CudaRender {

public:
	WorldRender(const std::string & filename);
	virtual ~WorldRender();
	
	virtual void setBufferSize(const int & w, const int & h);
	virtual void render();
	
};

}
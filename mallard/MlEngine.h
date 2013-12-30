/*
 *  MlEngine.h
 *  mallard
 *
 *  Created by jian zhang on 12/31/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <RenderEngine.h>
class MlEngine : public RenderEngine {
public:
	MlEngine();
	virtual ~MlEngine();
	
	virtual void render();
protected:

private:
	
};
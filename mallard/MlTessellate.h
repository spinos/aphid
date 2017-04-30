/*
 *  MlTessellate.h
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseTessellator.h>

class MlFeather;
class MlTessellate : public BaseTessellator {
public:
	MlTessellate();
	virtual ~MlTessellate();
	virtual void cleanup();
	void setFeather(MlFeather * feather);
	void evaluate(MlFeather * feather);

private:
	void initIndices();
	unsigned m_numSegment;
};
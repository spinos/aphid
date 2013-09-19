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
	void setFeather(MlFeather * feather);
	void evaluate(const MlFeather * feather);
	void createVertices(const MlFeather * feather);
	void createIndices(const MlFeather * feather);
private:
	MlFeather * m_feather;
};
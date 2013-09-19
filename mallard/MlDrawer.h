/*
 *  MlDrawer.h
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BezierDrawer.h>
class MlSkin;
class MlCalamus;
class MlTessellate;
class MlDrawer : public BezierDrawer {
public:
	MlDrawer();
	virtual ~MlDrawer();
	void drawFeather(MlSkin * skin) const;
	void drawAFeather(MlSkin * skin, MlCalamus * c) const;
private:
	MlTessellate * m_featherTess;
};
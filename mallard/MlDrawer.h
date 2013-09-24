/*
 *  MlDrawer.h
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseDrawer.h>
#include <DrawBuffer.h>
class MlSkin;
class MlCalamus;
class MlTessellate;
class MlDrawer : public BaseDrawer, public DrawBuffer {
public:
	MlDrawer();
	virtual ~MlDrawer();
	void drawFeather(MlSkin * skin) const;
	void drawAFeather(MlSkin * skin, MlCalamus * c) const;
	void hideAFeather(MlCalamus * c);
	void hideActive(MlSkin * skin);
	virtual void rebuildBuffer(MlSkin * skin);
private:
	MlTessellate * m_featherTess;
};
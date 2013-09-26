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
#include <BlockDrawBuffer.h>
class MlSkin;
class MlCalamus;
class MlTessellate;
class MlDrawer : public BaseDrawer, public BlockDrawBuffer {
public:
	MlDrawer();
	virtual ~MlDrawer();
	void drawFeather(MlSkin * skin) const;
	void hideAFeather(MlCalamus * c);
	void hideActive(MlSkin * skin);
	void updateActive(MlSkin * skin);
	void computeAFeather(MlSkin * skin, MlCalamus * c);
	void addToBuffer(MlSkin * skin);
private:
	void tessellate(MlSkin * skin, MlCalamus * c);
private:
	MlTessellate * m_featherTess;
};
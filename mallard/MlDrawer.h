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
#include <CacheFile.h>
class MlSkin;
class MlCalamus;
class MlTessellate;
class MlFeather;
class MlDrawer : public BaseDrawer, public BlockDrawBuffer, public CacheFile {
public:
	MlDrawer();
	virtual ~MlDrawer();
	void drawFeather(MlSkin * skin) const;
	void hideAFeather(MlCalamus * c);
	void hideActive(MlSkin * skin);
	void updateActive(MlSkin * skin);
	void updateBuffer(MlCalamus * c);
	void addToBuffer(MlSkin * skin);
	void rebuildBuffer(MlSkin * skin);
	void setCurrentFrame(int x);
private:
    void computeFeather(MlSkin * skin, MlCalamus * c);
	void tessellate(MlFeather * f);
	void writeToCache(MlSkin * skin, const std::string & sliceName);
	void readFromCache(MlSkin * skin, const std::string & sliceName);
private:
	MlTessellate * m_featherTess;
	int m_currentFrame;
};
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
#include <MlCache.h>

class MlSkin;
class MlCalamus;
class MlTessellate;
class MlFeather;
class MlDrawer : public BaseDrawer, public BlockDrawBuffer, public MlCache {
public:
	MlDrawer();
	virtual ~MlDrawer();
	void draw(MlSkin * skin) const;
	void hideAFeather(MlCalamus * c);
	void hideActive(MlSkin * skin);
	void updateActive(MlSkin * skin);
	void addToBuffer(MlSkin * skin);
	void computeBufferIndirection(MlSkin * skin);
	void readBuffer(MlSkin * skin);
	void rebuildBuffer(MlSkin * skin, bool forced = false);
	void setCurrentFrame(int x);
	void setCurrentOrigin(const Vector3F & at);
	
protected:
	
private:
	void updateBuffer(MlCalamus * c);
    void computeFeather(MlSkin * skin, MlCalamus * c);
	void computeFeather(MlSkin * skin, MlCalamus * c, const Vector3F & p, const Matrix33F & space);
	void tessellate(MlFeather * f);
	void writeToCache(const std::string & sliceName);
	void readFromCache(const std::string & sliceName);
private:
    Vector3F m_currentOrigin;
	MlTessellate * m_featherTess;
	MlSkin * skin;
	int m_currentFrame;
};
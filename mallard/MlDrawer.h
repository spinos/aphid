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
#include <BarbWorks.h>

class MlSkin;
class MlCalamus;
class MlTessellate;
class MlFeather;
class MlDrawer : public BaseDrawer, public BlockDrawBuffer, public BarbWorks {
public:
	MlDrawer();
	virtual ~MlDrawer();
	void draw() const;
	void hideAFeather(MlCalamus * c);
	void hideActive();
	void updateActive();
	void addToBuffer();
	void computeBufferIndirection();
	void readBuffer();
	void rebuildBuffer();
	void setCurrentFrame(int x);
	void setCurrentOrigin(const Vector3F & at);
	
	int currentFrame() const;
protected:
	
private:
	void updateBuffer(MlCalamus * c);
	void tessellate(MlFeather * f);
	void writeToCache(const std::string & sliceName);
	void readFromCache(const std::string & sliceName);
	void rebuildIgnoreCache();
private:
    Vector3F m_currentOrigin;
	MlTessellate * m_featherTess;
	int m_currentFrame;
};
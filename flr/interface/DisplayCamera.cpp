/*
 *  DisplayCamera.cpp
 *  
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DisplayCamera.h"
#include "BufferBlock.h"
#include <math/BaseCamera.h>

using namespace aphid;

DisplayCamera::DisplayCamera()
{
	m_lastViewFrame = new Matrix44F(0.f);
}

DisplayCamera::~DisplayCamera()
{}

void DisplayCamera::setCamera(aphid::BaseCamera* x)
{ m_camera = x; }

void DisplayCamera::setFrameCenter(int x, int y)
{ 
	m_frameCenterX = x;
	m_frameCenterY = y;
	m_deltaX = 1.f / (float)x;
	m_deltaY = 1.f / (float)y;
}

bool DisplayCamera::isViewFrameChanged() const
{ return !(m_lastViewFrame->isEqual(m_camera->fSpace) ); }

void DisplayCamera::updateViewFrame()
{ m_lastViewFrame->copy(m_camera->fSpace); }

void DisplayCamera::setBlockView(BufferBlock* blk)
{
	const int& tx0 = blk->tileX();
	const int tx1 = tx0 + BufferBlock::BlockSize();
	const int& ty0 = blk->tileY();
	const int ty1 = ty0 + BufferBlock::BlockSize();
	Vector3F ori, dir;
	m_camera->incidentRay(tx0, ty0, ori, dir);
	blk->setFrame(0, (const float*)&ori, (const float*)&dir);
	m_camera->incidentRay(tx1, ty0, ori, dir);
	blk->setFrame(1, (const float*)&ori, (const float*)&dir);
	m_camera->incidentRay(tx0, ty1, ori, dir);
	blk->setFrame(2, (const float*)&ori, (const float*)&dir);
	m_camera->incidentRay(tx1, ty1, ori, dir);
	blk->setFrame(3, (const float*)&ori, (const float*)&dir);

}

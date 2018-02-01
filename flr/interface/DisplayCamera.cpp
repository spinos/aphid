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

DisplayCamera::DisplayCamera() :
m_isChanged(false)
{
	m_lastViewFrame = new Matrix44F(0.f);
}

DisplayCamera::~DisplayCamera()
{}

void DisplayCamera::setCamera(aphid::BaseCamera* x)
{ m_camera = x; }

void DisplayCamera::setChanged()
{ m_isChanged = true; }

bool DisplayCamera::isChanged() const
{ return m_isChanged; }

void DisplayCamera::updateViewFrame()
{ 
#if 0
	std::cout<<"\n update camera tm"<<m_camera->fSpace
			<<"\n fov"<<m_camera->fieldOfView()
			<<"\n frm "<<m_camera->frameWidth()<<"x"<<m_camera->frameHeight()
			<<"\n port "<<m_camera->portWidth()<<"x"<<m_camera->portHeight();
#endif
	static const float cornerOffset[4][2] = {
	{-.5f, .5f},
	{ .5f, .5f},
	{-.5f, -.5f},
	{ .5f, -.5f}};
	
	for(int i=0;i<4;++i) {
		Vector3F corner(cornerOffset[i][0] * m_camera->frameWidth(), 
						cornerOffset[i][1] * m_camera->frameHeight(), 
						-m_camera->nearClipPlane() );	
		corner = m_camera->fSpace.transform(corner);
		memcpy(m_framePoints[i], &corner, 12);
#if 0		
		std::cout<<"\n corner"<<i<<"("<<m_framePoints[i][0]
					<<","<<m_framePoints[i][1]
					<<","<<m_framePoints[i][2]<<")";
#endif
	}
	
	m_deltaX = 1.f / (float)m_camera->portWidth();
	m_deltaY = 1.f / (float)m_camera->portHeight();
	
	m_lastViewFrame->copy(m_camera->fSpace); 
	m_isChanged = false;
}

void DisplayCamera::setBlockView(BufferBlock* blk) const
{
	const int& tx0 = blk->tileX();
	const int tx1 = tx0 + BufferBlock::BlockSize();
	const int& ty0 = blk->tileY();
	const int ty1 = ty0 + BufferBlock::BlockSize();
	
	const Vector3F peye = m_camera->eyePosition();
	
	Vector3F ori, dir;
	getPointOnFrame((float* )&ori, tx0, ty0);
	dir = ori - peye;
	dir.normalize();
	
	blk->setFrame(0, (const float*)&ori, (const float*)&dir);
	
	getPointOnFrame((float* )&ori, tx1, ty0);
	dir = ori - peye;
	dir.normalize();
	
	blk->setFrame(1, (const float*)&ori, (const float*)&dir);
	
	getPointOnFrame((float* )&ori, tx0, ty1);
	dir = ori - peye;
	dir.normalize();

	blk->setFrame(2, (const float*)&ori, (const float*)&dir);
	
	getPointOnFrame((float* )&ori, tx1, ty1);
	dir = ori - peye;
	dir.normalize();
	
	blk->setFrame(3, (const float*)&ori, (const float*)&dir);

}

void DisplayCamera::getPointOnFrame(float* pnt, const int& px, const int& py) const
{
	float alpha = m_deltaX * px;
	float mx[2][3];
	mx[0][0] = (1.f - alpha) * m_framePoints[0][0] + alpha * m_framePoints[1][0];
	mx[0][1] = (1.f - alpha) * m_framePoints[0][1] + alpha * m_framePoints[1][1];
	mx[0][2] = (1.f - alpha) * m_framePoints[0][2] + alpha * m_framePoints[1][2];
	
	mx[1][0] = (1.f - alpha) * m_framePoints[2][0] + alpha * m_framePoints[3][0];
	mx[1][1] = (1.f - alpha) * m_framePoints[2][1] + alpha * m_framePoints[3][1];
	mx[1][2] = (1.f - alpha) * m_framePoints[2][2] + alpha * m_framePoints[3][2];
	
	float beta = m_deltaY * py;
	pnt[0] = (1.f - beta) * mx[0][0] + beta * mx[1][0];
	pnt[1] = (1.f - beta) * mx[0][1] + beta * mx[1][1];
	pnt[2] = (1.f - beta) * mx[0][2] + beta * mx[1][2];
	
}

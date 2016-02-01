/*
 *  ViewCull.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ViewCull.h"

ViewCull::ViewCull() : m_enabled(false) {}
ViewCull::~ViewCull() {}
	
void ViewCull::enable()
{ m_enabled = true; }

void ViewCull::disable()
{ m_enabled = false; }

void ViewCull::setFrustum(const float & horizontalApeture,
			const float & verticalApeture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar)
{
	m_frustum.set(horizontalApeture, 
				verticalApeture,
				focalLength,
				clipNear,
				clipFar,
				m_space);
}
	
Matrix44F *	ViewCull::cameraSpaceP()
{ return &m_space; }

const Matrix44F & ViewCull::cameraSpace() const
{ return m_space; }

const AFrustum & ViewCull::frustum() const
{ return m_frustum; }

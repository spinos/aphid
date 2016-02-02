/*
 *  ViewCull.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ViewCull.h"
#include <GjkIntersection.h>

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

Matrix44F * ViewCull::cameraInvSpaceP()
{ return &m_invSpace; }

const Matrix44F & ViewCull::cameraSpace() const
{ return m_space; }

const AFrustum & ViewCull::frustum() const
{ return m_frustum; }

bool ViewCull::cullByFrustum(const Vector3F & center, const float & radius) const
{
	gjk::Sphere B(center, radius );
	if( gjk::Intersect1<AFrustum, gjk::Sphere>::Evaluate(m_frustum, B) )
		return false;
	return true;
}
//:~
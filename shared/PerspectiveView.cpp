/*
 *  PerspectiveView.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/3/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "PerspectiveView.h"

namespace aphid {

PerspectiveView::PerspectiveView(const PerspectiveCamera * camera)
{ m_camera = camera; }

PerspectiveView::~PerspectiveView()
{}

void PerspectiveView::update()
{
	m_frustum.set(m_camera->nearClipPlane(),
					m_camera->farClipPlane(),
					1.f,
					1.f / m_camera->aspectRatio(),
					m_camera->fieldOfView(),
					m_camera->fSpace );
	m_frameBase = tan(m_camera->fieldOfView()/360.f * 3.1415927f);
}

const cvx::Frustum * PerspectiveView::frustum() const
{ return &m_frustum; }

bool PerspectiveView::isBoxVisible(const BoundingBox & b) const
{ return m_frustum.intersectBBox(b); }

float PerspectiveView::lodBox(const BoundingBox & b) const
{
	Vector3F q = m_camera->fInverseSpace.transform(b.center() );
	const float r = b.radius();
	const float ncp = m_camera->nearClipPlane();
	float d = -q.z - r;
	if(d < ncp)
		return 1.f;
	
	const float fcp = m_camera->farClipPlane();
	
	d = (d - ncp ) / (fcp - ncp );
	d = m_frameBase * (ncp * (1.f - d) + fcp * d);
	
	return r / d; 
}

}
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

namespace aphid {

BaseView::BaseView() {}
BaseView::~BaseView() {}

void BaseView::setFrustum(const float & horizontalApeture,
			const float & verticalApeture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar)
{
	m_hfov = horizontalApeture * 0.5f / ( focalLength * 0.03937f );
	m_aspectRatio = verticalApeture / horizontalApeture;
	m_farClip = clipFar;
	m_frustum.set(m_hfov, 
				m_aspectRatio,
				clipNear,
				clipFar,
				m_space);
}

void BaseView::setOrthoFrustum(const float & orthoWidth,
			const float & orthoHeight,
			const float & clipNear,
			const float & clipFar)
{
	m_hfov = -1.f;
	m_aspectRatio = orthoHeight / orthoWidth;
	m_farClip = clipFar;
	m_frustum.setOrtho(orthoWidth * .5f, 
				m_aspectRatio,
				clipNear,
				clipFar,
				m_space);	
}

void BaseView::updateAspectRatio(const int & w, const int & h)
{
	m_aspectRatio = (float)h / (float)w;
	m_frustum.set(m_hfov, 
				m_aspectRatio,
				-1.f,
				m_farClip,
				m_space);
}
	
Matrix44F *	BaseView::cameraSpaceP()
{ return &m_space; }

Matrix44F * BaseView::cameraInvSpaceP()
{ return &m_invSpace; }

const Matrix44F & BaseView::cameraSpace() const
{ return m_space; }

const Matrix44F & BaseView::cameraInvSpace() const
{ return m_invSpace; }

const AFrustum & BaseView::frustum() const
{ return m_frustum; }

bool BaseView::isPerspective() const
{ return m_hfov > 0.f; }

const float & BaseView::farClipPlane() const
{ return m_farClip; }

const Vector3F & BaseView::eyePosition() const
{ return m_eyePosition; }

void BaseView::setEyePosition(float * p)
{ m_eyePosition.set(p[0], p[1], p[2]); }

const float & BaseView::aspectRatio() const
{ return m_aspectRatio; }

const float & BaseView::hfov() const
{ return m_hfov; }

void BaseView::setRect(const int & x, const int & y)
{ 
	m_rect.set(x, y); 
	updateAspectRatio(x, y);
}

const RectangleI & BaseView::rect() const
{ return m_rect; }

void BaseView::setSubRect(const int & x0, const int & y0, const int & x1, const int & y1)
{ m_subRect.set(x0, y0, x1, y1); }

const RectangleI & BaseView::subRect() const
{ return m_subRect; }

const int BaseView::numPixels() const
{ return m_rect.area(); }

ViewCull::ViewCull() : m_enabled(false), m_portAspectRatio(1.f) {}
ViewCull::~ViewCull() {}
	
void ViewCull::enableView()
{ m_enabled = true; }

void ViewCull::disableView()
{ m_enabled = false; }

bool ViewCull::cullByFrustum(const Vector3F & center, const float & radius) const
{
	gjk::Sphere B;
	B.set(center, radius );
	if( gjk::Intersect1<AFrustum, gjk::Sphere>::Evaluate(frustum(), B) )
		return false;
	return true;
}

bool ViewCull::cullByFrustum(const BoundingBox & box) const
{
    if( gjk::Intersect1<AFrustum, BoundingBox>::Evaluate(frustum(), box) )
		return false;
    return true; 
}

/// in viewport
void ViewCull::ndc(const Vector3F & cameraP, float & coordx, float & coordy) const
{
	float d = -cameraP.z;
	if(d<1.f) d= 1.f; 
	float h_max = d * hfov();
	float h_min = -h_max;
	float v_max = h_max * aspectRatio();
	float v_min = -v_max;
	coordx = (cameraP.x/m_overscan - h_min) / (h_max - h_min);
	coordy = (cameraP.y/m_overscan/(m_portAspectRatio / aspectRatio() ) - v_min) / (v_max - v_min);
	
	if(coordx < 0.f) coordx = 0.f;
	if(coordx > .999f) coordx = .999f;
	if(coordy < 0.f) coordy = 0.f;
	if(coordy > .999f) coordy = .999f;
}

bool ViewCull::cullByLod(const float & localZ, const float & radius,
					const float & lowLod, const float & highLod,
					float & details) const
{
	details = radius / (m_detailWidth * localZ / farClipPlane() );
	if(details > .999f) details = .999f;
	if(details < lowLod || details >= highLod) return true;
	return false;
}

float ViewCull::cameraDepth(const Vector3F & p) const
{ return cameraInvSpace().transform(p).z; }

void ViewCull::getFarClipDepth(float & clip, const BoundingBox & b) const
{
    int i = 0;
    for(;i<8;++i) {
        float d = cameraDepth(b.X(i) );
        if(clip > d) clip = d;
    }
}

const bool & ViewCull::hasView() const
{ return m_enabled; }
	
void ViewCull::setOverscan(const double & x)
{ m_overscan = x; }

void ViewCull::setViewportAspect(const int & portWidth,
					const int & portHeight)
{
	if(portWidth>0) m_portAspectRatio = (float)portHeight/(float)portWidth;
}

const float & ViewCull::overscan() const
{ return m_overscan; }

void ViewCull::setFrustum(const float & horizontalApeture,
			const float & verticalApeture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar)
{
	BaseView::setFrustum(horizontalApeture, verticalApeture, 
					focalLength, clipNear, clipFar);
/// 1 / 30 of port width
	m_detailWidth = -clipFar * hfov() * .066f;
}

void ViewCull::setOrthoFrustum(const float & orthoWidth,
			const float & orthoHeight,
			const float & clipNear,
			const float & clipFar)
{
	BaseView::setOrthoFrustum(orthoWidth, orthoHeight,
					clipNear, clipFar);
	m_detailWidth = orthoWidth * .066f;
}

}
//:~
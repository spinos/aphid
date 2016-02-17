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
	
void ViewCull::enableView()
{ m_enabled = true; }

void ViewCull::disableView()
{ m_enabled = false; }

void ViewCull::setFrustum(const float & horizontalApeture,
			const float & verticalApeture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar)
{
	m_hfov = horizontalApeture * 0.5f / ( focalLength * 0.03937f );
	m_aspectRatio = verticalApeture / horizontalApeture;
	m_farClip = clipFar;
/// 1 / 30 of port width
	m_detailWidth = -clipFar * m_hfov * .066f;
	m_frustum.set(m_hfov, 
				m_aspectRatio,
				clipNear,
				clipFar,
				m_space);
}

void ViewCull::setOrthoFrustum(const float & orthoWidth,
			const float & aspectRatio,
			const float & clipNear,
			const float & clipFar)
{
	m_hfov = -1.f;
	m_aspectRatio = aspectRatio;
	m_farClip = clipFar;
	m_detailWidth = orthoWidth * .05f;
	m_frustum.setOrtho(orthoWidth * .5f, 
				m_aspectRatio,
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

const Matrix44F & ViewCull::cameraInvSpace() const
{ return m_invSpace; }

const AFrustum & ViewCull::frustum() const
{ return m_frustum; }

bool ViewCull::cullByFrustum(const Vector3F & center, const float & radius) const
{
	gjk::Sphere B(center, radius );
	if( gjk::Intersect1<AFrustum, gjk::Sphere>::Evaluate(m_frustum, B) )
		return false;
	return true;
}

bool ViewCull::cullByFrustum(const BoundingBox & box) const
{
    if( gjk::Intersect1<AFrustum, BoundingBox>::Evaluate(m_frustum, box) )
		return false;
    return true; 
}

/// in viewport
void ViewCull::ndc(const Vector3F & cameraP, float & coordx, float & coordy) const
{
	float d = -cameraP.z;
	if(d<1.f) d= 1.f; 
	float h_max = d * m_hfov;
	float h_min = -h_max;
	float v_max = h_max * m_aspectRatio;
	float v_min = -v_max;
	coordx = (cameraP.x/m_overscan - h_min) / (h_max - h_min);
	coordy = (cameraP.y/m_overscan/(m_portAspectRatio / m_aspectRatio) - v_min) / (v_max - v_min);
	
	if(coordx < 0.f) coordx = 0.f;
	if(coordx > .999f) coordx = .999f;
	if(coordy < 0.f) coordy = 0.f;
	if(coordy > .999f) coordy = .999f;
}

bool ViewCull::cullByLod(const float & localZ, const float & radius,
					const float & lowLod, const float & highLod,
					float & details) const
{
	details = radius / (m_detailWidth * localZ / m_farClip);
	if(details > .999f) details = .999f;
	if(details < lowLod || details >= highLod) return true;
	return false;
}

bool ViewCull::isPerspective() const
{ return m_hfov > 0.f; }

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

const float ViewCull::nearClipPlane() const
{ return 1.f; }

const float & ViewCull::farClipPlane() const
{ return m_farClip; }

const float & ViewCull::hfov() const
{ return m_hfov; }
	
void ViewCull::setViewport(const double & overscan,
					const int & portWidth,
					const int & portHeight)
{
	m_overscan = overscan;
	m_portAspectRatio = (float)portHeight/(float)portWidth;
}

const float & ViewCull::overscan() const
{ return m_overscan; }
//:~
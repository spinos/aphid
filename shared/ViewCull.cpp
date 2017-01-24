/*
 *  ViewCull.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *  reference http://paulbourke.net/miscellaneous/lens/
 */

#include "ViewCull.h"
#include <GjkIntersection.h>
#include <cmath>

namespace aphid {

BaseView::BaseView() 
{
	m_centerOfInterest.set(0.f, 0.f, 0.f);
	m_space.setIdentity();
	m_space.setTranslation(Vector3F(0.f, 0.f, 100.f) );
	m_invSpace.setIdentity();
	m_invSpace.setTranslation(Vector3F(0.f, 0.f, -100.f) );
	
/// 35mm Academy
	std::cout<<"\n angle of view "<<180.f/3.14f*2.f * atan(21.9456f/2.f/35.f)<<" deg";
	setFrustum(.864f, .63f, 35.f, -1.f, -20000.f);
	
}

BaseView::~BaseView() {}

void BaseView::tumble(int dx, int dy, int portWidth)
{
	Vector3F side  = m_space.getSide();
	Vector3F up    = m_space.getUp();
	Vector3F front = m_space.getFront();
	Vector3F eye = m_space.getTranslation();	
	Vector3F toEye = eye - m_centerOfInterest;
	float dist = toEye.length();
	const float scaleing = dist * 2.f / (float)portWidth;
	eye -= side * (float)dx * scaleing;
	eye += up * (float)dy * scaleing;
	
	toEye = eye - m_centerOfInterest;
	toEye.normalize();
	
	eye = m_centerOfInterest + toEye * dist;
	m_space.setTranslation(eye);
	
	front = toEye;
	
	side = up.cross(front);
	side.y = 0.f;
	side.normalize();
	
	up = front.cross(side);
	up.normalize();
	
	m_space.setOrientations(side, up, front);
	
	m_invSpace = m_space;
	m_invSpace.inverse();
}

void BaseView::track(int dx, int dy, int portWidth)
{
	Vector3F side  = m_space.getSide();
	Vector3F up    = m_space.getUp();
	Vector3F eye = m_space.getTranslation();
	
	const float scaling = perspectivity(portWidth);
	
	side *= (float)dx * scaling;
	up *= (float)dy * scaling;
	eye -= side;
	eye += up;
	
	m_centerOfInterest -= side;
	m_centerOfInterest += up;
	
	m_space.setTranslation(eye);
	
	m_invSpace = m_space;
	m_invSpace.inverse();
}

void BaseView::zoom(int dz, int portWidth)
{
	Vector3F front = m_space.getFront();
	Vector3F eye = m_space.getTranslation();
	Vector3F toEye = eye - m_centerOfInterest;
	const float dist = toEye.length();
	
	const float fra = (float)dz/(float)portWidth * 7.f;
	
	eye += front * dist * -fra;
	if(fra > 0.f && dist < 10.f) {
		m_centerOfInterest += front * dist * -fra * 0.1f;
	}
	
	m_space.setTranslation(eye);
	m_invSpace = m_space;
	m_invSpace.inverse();
}

float BaseView::perspectivity(int portWidth) const
{ return m_hfov * 2.f * eyePosition().distanceTo(m_centerOfInterest) / (float)portWidth; }

void BaseView::frameAll()
{
    m_centerOfInterest.setZero();
    m_space.setIdentity();
	m_space.setTranslation(Vector3F(0.f, 0.f, 100.f) );
	m_invSpace.setIdentity();
	m_invSpace.setTranslation(Vector3F(0.f, 0.f, -100.f) );
}

void BaseView::setFrustum(const float & horizontalAperture,
			const float & verticalAperture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar)
{
/// 0.03937f is millimeter to inch conversion
	m_hfov = horizontalAperture * 0.5f / ( focalLength * 0.03937f );
	m_aspectRatio = verticalAperture / horizontalAperture;
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
	
Matrix44F *	BaseView::cameraSpaceR()
{ return &m_space; }

Matrix44F * BaseView::cameraInvSpaceR()
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

Vector3F BaseView::eyePosition() const
{ return m_space.getTranslation(); }

void BaseView::setEyePosition(const Vector3F & p)
{ m_space.setTranslation(p); }

void BaseView::setCenterOfInterest(const Vector3F & p)
{ m_centerOfInterest = p; }

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

void BaseView::updateRayFrameVec()
{ frustum().toRayFrame(m_rayFrameVec, rect().width(), rect().height() ); }

Vector3F * BaseView::rayFrameVec()
{ return m_rayFrameVec; }

void BaseView::frameAll(const BoundingBox & b)
{
	Vector3F eye = b.center();
	eye.z = b.getMax(2) + b.distance(0) / hfov() * .55f + 120.f;
	setEyePosition(eye);
	
	Matrix44F m;
	m.setTranslation(eye);
	*cameraSpaceR() = m;
	m.inverse();
	*cameraInvSpaceR() = m;
	setFrustum(1.33f, 1.f, 26.2f, -1.f, -1000.f);
}

Vector3F BaseView::directionToEye() const
{
    Vector3F v = eyePosition() - m_centerOfInterest;
    v.normalize();
    return v;
}

void BaseView::setFarClip(const float & x)
{ m_farClip = x; }

void BaseView::updateInvSpace()
{
    m_invSpace = m_space;
	m_invSpace.inverse();
}

void BaseView::updateFrustum()
{ setFrustum(.864f, .63f, 35.f, -1.f, m_farClip); }

float BaseView::cameraDepth(const Vector3F & p) const
{ return cameraInvSpace().transform(p).z; }

float BaseView::relativeSizeAtDepth(const Vector3F & p, 
				const float & w) const
{
	if(!isPerspective() ) {
		return frustum().X(0).distanceTo(frustum().X(1) ) * .5f * w; 
	}
	
	const float d = cameraDepth(p);
	if(d > -1e-2f) {
		return 1.f;
	}
	
	return d / farClipPlane() * hfov() * (frustum().X(4).distanceTo(frustum().X(5) ) ) * w;
}

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
	if(details < 0.f) {
		return false;
	}
	if(details > .999f) {
		details = .999f;
	}
	return (details < lowLod || details >= highLod);
}

void ViewCull::getFarClipDepth(float & clip, const BoundingBox & b) const
{
    int i = 0;
    for(;i<8;++i) {
        float d = cameraDepth(b.X(i) );
        if(clip > d) clip = d;
    }
/// truncate far large value
	if(clip < -1e7f) {
		std::cout<<"\n truncate large far clip "<<clip<<" to 1e7f";
		clip = -1e7f;
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
	m_detailWidth = hfov() * (frustum().X(4).distanceTo(frustum().X(5) ) ) * .067f;
}

void ViewCull::setOrthoFrustum(const float & orthoWidth,
			const float & orthoHeight,
			const float & clipNear,
			const float & clipFar)
{
	BaseView::setOrthoFrustum(orthoWidth, orthoHeight,
					clipNear, clipFar);
	m_detailWidth = orthoWidth * .067f;
}

}
//:~
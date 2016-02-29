/*
 *  ViewCull.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AFrustum.h>
#include <BoundingBox.h>
#include <BoundingRectangle.h>

namespace aphid {
    
class BaseView {

	AFrustum m_frustum;
	Matrix44F m_space, m_invSpace;
	RectangleI m_rect, m_subRect;
	Vector3F m_eyePosition;
	float m_hfov, m_aspectRatio, m_farClip; 
	
public:
    BaseView();
    virtual ~BaseView();
    
	bool isPerspective() const;
			
	const Matrix44F & cameraSpace() const;
	const Matrix44F & cameraInvSpace() const;
	const AFrustum & frustum() const;
	
	const float & farClipPlane() const;
	const Vector3F & eyePosition() const;
	const float & aspectRatio() const;
	const RectangleI & Rect() const;
	const RectangleI & subRect() const;
	
protected:
	void setRect(const int & x, const int & y);
    void setSubRect(const int & x0, const int & y0, const int & x1, const int & y1);
    void setEyePosition(float * p);
	Matrix44F *	cameraSpaceP();
	Matrix44F * cameraInvSpaceP();
	
	const float & hfov() const;
	
/// cliping is negative in camera space
	virtual void setFrustum(const float & horizontalApeture,
			const float & verticalApeture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar);
			
	virtual void setOrthoFrustum(const float & orthoWidth,
			const float & orthoHeight,
			const float & clipNear,
			const float & clipFar);
private:
    
};

class ViewCull : public BaseView {
	
	float m_detailWidth, m_overscan, m_portAspectRatio;
	bool m_enabled;
	
public:
	ViewCull();
	virtual ~ViewCull();
	
	void enableView();
	void disableView();
	const bool & hasView() const;
	
	bool cullByFrustum(const Vector3F & center, const float & radius) const;
	bool cullByFrustum(const BoundingBox & box) const;
	bool cullByLod(const float & localZ, const float & radius,
					const float & lowLod, const float & highLod,
					float & details) const;
	
	float cameraDepth(const Vector3F & p) const;
	void getFarClipDepth(float & clip, const BoundingBox & b) const;
    
protected:
	virtual void setFrustum(const float & horizontalApeture,
			const float & verticalApeture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar);
			
	virtual void setOrthoFrustum(const float & orthoWidth,
			const float & orthoHeight,
			const float & clipNear,
			const float & clipFar);
			
	void ndc(const Vector3F & cameraP, float & coordx, float & coordy) const;
	void setOverscan(const double & x);
	void setViewportAspect(const int & portWidth,
					const int & portHeight);
	const float & overscan() const;
	
private:

};

}
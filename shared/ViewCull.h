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
#include <math/BoundingBox.h>
#include <math/BoundingRectangle.h>

namespace aphid {
    
class BaseView {

	AFrustum m_frustum;
	Matrix44F m_space, m_invSpace;
	RectangleI m_rect, m_subRect;
	Vector3F m_rayFrameVec[6];
	Vector3F m_centerOfInterest;
	float m_hfov, m_aspectRatio, m_farClip; 
	
public:
    BaseView();
    virtual ~BaseView();
    
	void tumble(int dx, int dy, int portWidth);
	void track(int dx, int dy, int portWidth);
	void zoom(int dz, int portWidth);
	void updateRayFrameVec();
	virtual void frameAll();
	
	bool isPerspective() const;
			
	const Matrix44F & cameraSpace() const;
	const Matrix44F & cameraInvSpace() const;
	const AFrustum & frustum() const;
	
	const float & farClipPlane() const;
	Vector3F eyePosition() const;
	const float & aspectRatio() const;
	const RectangleI & rect() const;
	const RectangleI & subRect() const;
	const int numPixels() const;
	
protected:
	void setRect(const int & x, const int & y);
    void setSubRect(const int & x0, const int & y0, const int & x1, const int & y1);
    void setEyePosition(const Vector3F & p);
	void setCenterOfInterest(const Vector3F & p);
	void setFarClip(const float & x);
	Matrix44F *	cameraSpaceR();
	Matrix44F * cameraInvSpaceR();
	
	const float & hfov() const;
	
/// cliping is negative in camera space
/// camera apertures are in inches
/// focal length is in millimeters
	virtual void setFrustum(const float & horizontalAperture,
			const float & verticalAperture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar);
			
	virtual void setOrthoFrustum(const float & orthoWidth,
			const float & orthoHeight,
			const float & clipNear,
			const float & clipFar);
			
	void updateAspectRatio(const int & w, const int & h);
	
	Vector3F * rayFrameVec();
	
	void frameAll(const BoundingBox & b);
	float perspectivity(int portWidth) const;
	Vector3F directionToEye() const;
	void updateInvSpace();
	void updateFrustum();
	
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
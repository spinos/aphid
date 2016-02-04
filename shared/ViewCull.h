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

class ViewCull {
	
	AFrustum m_frustum;
	Matrix44F m_space, m_invSpace;
	float m_hfov, m_aspectRatio, m_farClip, m_detailWidth;
	bool m_enabled;
	
public:
	ViewCull();
	virtual ~ViewCull();
	
	void enable();
	void disable();
	bool isPerspective() const;
	
/// cliping is negative in camera space
	void setFrustum(const float & horizontalApeture,
			const float & verticalApeture,
			const float & focalLength,
			const float & clipNear,
			const float & clipFar);
			
	void setOrthoFrustum(const float & orthoWidth,
			const float & aspectRatio,
			const float & clipNear,
			const float & clipFar);
			
	Matrix44F *	cameraSpaceP();
	Matrix44F * cameraInvSpaceP();
	const Matrix44F & cameraSpace() const;
	const Matrix44F & cameraInvSpace() const;
	const AFrustum & frustum() const;
	
	bool cullByFrustum(const Vector3F & center, const float & radius) const;
	bool cullByFrustum(const BoundingBox & box) const;
	bool cullByLod(const float & localZ, const float & radius,
					const float & lowLod, const float & highLod) const;
					
protected:
	void ndc(const Vector3F & cameraP, float & coordx, float & coordy) const;
	
private:

};
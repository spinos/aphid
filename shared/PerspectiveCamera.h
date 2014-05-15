/*
 *  PerspectiveCamera.h
 *  fit
 *
 *  Created by jian zhang on 4/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BaseCamera.h>
class PerspectiveCamera : public BaseCamera {
public:
	PerspectiveCamera();
	virtual ~PerspectiveCamera();
	
	virtual bool isOrthographic() const;
	virtual float fieldOfView() const;
	virtual float frameWidth() const;
	virtual float frameWidthRel() const;
	virtual void zoom(int y);
	virtual void incidentRay(int x, int y, Vector3F & origin, Vector3F & worldVec) const;
	virtual void setFieldOfView(float x);
	virtual void screenToWorldVectorAt(int x, int y, float depth, Vector3F & worldVec) const;
private:
	float m_fov;
};

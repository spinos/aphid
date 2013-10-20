/*
 *  TransformManipulator.h
 *  aphid
 *
 *  Created by jian zhang on 10/19/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BaseTransform.h>
#include <Ray.h>
#include <ToolContext.h>
#include <Plane.h>
class TransformManipulator : public BaseTransform {
public:
	TransformManipulator();
	virtual ~TransformManipulator();
	
	void attachTo(BaseTransform * subject);
	void reattach();
	void detach();
	
	bool isDetached() const;
	
	void start(const Ray * r);
	void perform(const Ray * r);
	void stop();
	
	void move(const Vector3F & d);
	void spin(const Vector3F & d);
	
	void setToMove();
	void setToRotate();
	ToolContext::InteractMode mode() const;
	
	void setRotateAxis(RotateAxis axis);
	RotateAxis rotateAxis() const;
	
	Vector3F hitPlaneNormal() const;
	virtual Vector3F rotatePlane(RotateAxis a) const;
	virtual Vector3F rotationBaseAngles() const;
	
	Vector3F startPoint() const;
	Vector3F currentPoint() const;
	
	bool started() const;
	
	BaseTransform * subject() const;
private:
	
private:
	RotateAxis m_rotateAxis;
	Vector3F m_startPoint, m_currentPoint;
	BaseTransform * m_subject;
	ToolContext::InteractMode m_mode;
	bool m_started;
};
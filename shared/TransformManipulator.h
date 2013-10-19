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
class TransformManipulator : public BaseTransform {
public:
	enum RotateAxis {
		AX,
		AY,
		AZ
	};
	
	TransformManipulator();
	virtual ~TransformManipulator();
	
	void attachTo(BaseTransform * subject);
	void detach();
	
	bool isDetached() const;
	
	void start(const Ray * r);
	void perform(const Ray * r);
	
	void move(const Vector3F & d);
	void spin(const Vector3F & d);
	
	BaseTransform * origin() const;
	
	void setToMove();
	void setToRotate();
	ToolContext::InteractMode mode() const;
	
	void setRotateAxis(RotateAxis axis);
	RotateAxis rotateAxis() const;
	
	Vector3F hitPlaneNormal() const;
	Vector3F rotatePlaneNormal(RotateAxis a) const;
	Vector3F translatePlaneNormal() const;
private:
	
private:
	RotateAxis m_rotateAxis;
	Vector3F m_startPoint;
	BaseTransform * m_subject;
	BaseTransform * m_origin;
	ToolContext::InteractMode m_mode;
};
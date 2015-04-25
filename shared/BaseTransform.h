/*
 *  BaseTransform.h
 *  eulerRot
 *
 *  Created by jian zhang on 10/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <Geometry.h>
#include <NamedEntity.h>
#include <Ray.h>

class BaseTransform : public Geometry, public NamedEntity {
public:
	enum RotateAxis {
		AX,
		AY,
		AZ
	};
	BaseTransform(BaseTransform * parent = 0);
	virtual ~BaseTransform();
	
	void setParent(BaseTransform * parent);
	BaseTransform * parent() const;
	
	void translate(const Vector3F & v);
	void setTranslation(const Vector3F & v);
	Vector3F translation() const;
	
	void rotate(const Vector3F & v);
	void setRotationAngles(const Vector3F & v);

	Matrix33F rotation() const;
	Vector3F rotationAngles() const;
	
	void addChild(BaseTransform * child);
	unsigned numChildren() const;
	BaseTransform * child(unsigned idx) const;
	
	void parentSpace(Matrix44F & dst) const;
	Matrix44F space() const;
	Matrix44F worldSpace() const;
	
	bool intersect(const Ray & ray) const;
	
	Vector3F translatePlane(RotateAxis a) const;
	virtual Vector3F rotatePlane(RotateAxis a) const;
	
	virtual Vector3F rotationBaseAngles() const;
	
	void detachChild(unsigned idx);
	
	void setRotateDOF(const Float3 & dof);
	Float3 rotateDOF() const;
	
	virtual const Type type() const;
protected:
	
	
private:
	Vector3F m_translation, m_angles;
	BaseTransform * m_parent;
	std::vector<BaseTransform *> m_children;
	Float3 m_rotateDOF;
};
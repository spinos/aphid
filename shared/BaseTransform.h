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
#include <geom/Geometry.h>
#include <Boundary.h>
#include <foundation/NamedEntity.h>

namespace aphid {

class BaseTransform : public Geometry, public Boundary {
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

	Matrix33F orientation() const;
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
	
	void setRotationOrder(Matrix33F::RotateOrder x);
	Matrix33F::RotateOrder rotationOrder() const;
	
	void setScale(const Vector3F & a);
	Vector3F scale() const;
	
	void setRotatePivot(const Vector3F & p, const Vector3F & t);
	Vector3F rotatePivot() const;
	Vector3F rotatePivotTranslate() const;
	
	void setScalePivot(const Vector3F & p, const Vector3F & t);
	Vector3F scalePivot() const;
	Vector3F scalePivotTranslate() const;
	
	virtual const Type type() const;
protected:
	
	
private:
	Vector3F m_translation, m_angles, m_scale, m_rotatePivot, m_scalePivot, m_rotatePivotTranslate, m_scalePivotTranslate;
	BaseTransform * m_parent;
	std::vector<BaseTransform *> m_children;
	Float3 m_rotateDOF;
	Matrix33F::RotateOrder m_rotationOrder;
};

}
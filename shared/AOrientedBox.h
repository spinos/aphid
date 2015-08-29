/*
 *  AOrientedBox.h
 *  aphid
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <AllMath.h>
#include <Geometry.h>
class AOrientedBox : public Geometry {
public:
	AOrientedBox();
	virtual ~AOrientedBox();
	
	void setCenter(const Vector3F & p);
	void setOrientation(const Matrix33F & m);
	void setExtent(const Vector3F & p);
	
	Vector3F center() const;
	Matrix33F orientation() const;
	Vector3F extent() const;
	Vector3F majorPoint(bool low) const;
	Vector3F majorVector(bool low) const;
	void getBoxVertices(Vector3F * dst) const;
	
	virtual const Type type() const;
	virtual const BoundingBox calculateBBox() const;
protected:

private:
	Matrix33F m_orientation;
	Vector3F m_center;
	Vector3F m_extent;
};
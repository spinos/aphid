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

namespace aphid {

class AOrientedBox : public Geometry {
public:
	AOrientedBox();
	virtual ~AOrientedBox();
	
	void setCenter(const Vector3F & p);
	void setOrientation(const Matrix33F & m);
	void setExtent(const Vector3F & p);
	void set8DOPExtent(const float & x0, const float & x1,
						const float & y0, const float & y1);
	
	const Vector3F & center() const;
	const Matrix33F & orientation() const;
	const Vector3F & extent() const;
	const float * dopExtent() const;
	Vector3F majorPoint(bool low) const;
	Vector3F majorVector(bool low) const;
    Vector3F minorPoint(bool low) const;
	Vector3F minorVector(bool low) const;
	void getBoxVertices(Vector3F * dst) const;
	
	virtual const Type type() const;
	virtual const BoundingBox calculateBBox() const;
	void limitMinThickness(const float & x);
	
protected:

private:
	Matrix33F m_orientation;
	Vector3F m_center;
	Vector3F m_extent;
/// in xy
	float m_8DOPExtent[4];
};

class DOP8Builder {

	Vector3F m_vert[18];
/// lateral 8 top and bottom
	Vector3F m_nor[10];
	Vector3F m_facevert[84];
	Vector3F m_facenor[84];
	int m_tri[84];
	int m_ntri;
	
public:
	DOP8Builder();
	void build(const AOrientedBox & ob);
	const int & numTriangles() const;
	const Vector3F * vertex() const;
	const Vector3F * normal() const;
};

}
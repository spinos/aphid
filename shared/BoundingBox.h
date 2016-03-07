	/*
 *  BoundingBox.h
 *  kdtree
 *
 *  Created by jian zhang on 10/17/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Vector3F.h>
namespace aphid {

class Ray;
class Plane;
class BoundingBox {
public:
	BoundingBox();
	BoundingBox(const float & x0, const float & y0, const float & z0,
	            const float & x1, const float & y1, const float & z1);
	BoundingBox(const float * d);
	void reset();
	void setOne();
	void setMin(float x, int axis);
	void setMax(float x, int axis);
	void setMin(float x, float y, float z);
	void setMax(float x, float y, float z);
	void update(const Vector3F & p);
	void updateMin(const Vector3F & p);
	void updateMax(const Vector3F & p);
	
	const int getLongestAxis() const;
	const float getLongestDistance() const;
	const float getMin(int axis) const;
	const float getMax(int axis) const;
	const Vector3F getMin() const;
	const Vector3F getMax() const;
	const float area() const;
	const float volume() const;
	const float crossSectionArea(const int &axis) const;
	const float distance(const int &axis) const;
	const Vector3F normal(const int & i) const;
	
	void split(int axis, float pos, BoundingBox & left, BoundingBox & right) const;
	void expandBy(const BoundingBox &another);
	void expandBy(const Vector3F & pos);
	void expandBy(const Vector3F & pos, float r);
	void shrinkBy(const BoundingBox &another);
	void expand(float v);
	Vector3F center() const;
	const Vector3F corner(const int & i) const;
	char touch(const BoundingBox & b) const;
	
	float distanceTo(const Vector3F & pnt) const;
    float radius() const;
	float radiusXZ() const;
	
	char intersect(const BoundingBox & another) const;
	bool intersect(const BoundingBox & another, BoundingBox * tightBox) const;
	char intersect(const Ray &ray, float *hitt0, float *hitt1) const;
	bool intersect(const Plane & p, float & tmin, float & tmax) const;
	char isPointInside(const Vector3F & p) const;
	char isPointAround(const Vector3F & p, float threshold) const;
	char isBoxAround(const BoundingBox & b, float threshold) const;
	char inside(const BoundingBox & b) const;
	char isValid() const;
	void verbose() const;
	void verbose(const char * pref) const;
	const std::string str() const;
    friend std::ostream& operator<<(std::ostream &output, const BoundingBox & p) {
        output << p.str();
        return output;
    }
    int numPoints() const;
    Vector3F supportPoint(const Vector3F & v, Vector3F * localP = NULL) const;
    Vector3F X(int idx) const;
	const float * data() const;
	float m_data[6];
	int m_padding0, m_padding1;
};
}
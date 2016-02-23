/*
 *  AOrientedBox.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "AOrientedBox.h"

namespace aphid {

AOrientedBox::AOrientedBox() {}
AOrientedBox::~AOrientedBox() {}

void AOrientedBox::setCenter(const Vector3F & p)
{ m_center = p; }

void AOrientedBox::setOrientation(const Matrix33F & m)
{ m_orientation = m; }

void AOrientedBox::setExtent(const Vector3F & p)
{ m_extent = p; }

Vector3F AOrientedBox::center() const
{ return m_center; }

Matrix33F AOrientedBox::orientation() const
{ return m_orientation; }

Vector3F AOrientedBox::extent() const
{ return m_extent; }

Vector3F AOrientedBox::majorPoint(bool low) const
{ return (m_center + majorVector(low) * m_extent.x); }

Vector3F AOrientedBox::majorVector(bool low) const
{ 
	float dir = 1.f;
	if(low) dir = -1.f;
	return m_orientation.row(0) * dir;
}

Vector3F AOrientedBox::minorPoint(bool low) const
{ return (m_center + minorVector(low) * m_extent.y); }

Vector3F AOrientedBox::minorVector(bool low) const
{
    float dir = 1.f;
	if(low) dir = -1.f;
	return m_orientation.row(1) * dir;
}

void AOrientedBox::getBoxVertices(Vector3F * dst) const
{
	const Vector3F rx = m_orientation.row(0);
	const Vector3F ry = m_orientation.row(1);
	const Vector3F rz = m_orientation.row(2);
	const Vector3F eh = m_extent;
	dst[0] = m_center - rx * eh.x - ry * eh.y - rz * eh.z;
	dst[1] = m_center + rx * eh.x - ry * eh.y - rz * eh.z;
	dst[2] = m_center - rx * eh.x + ry * eh.y - rz * eh.z;
	dst[3] = m_center + rx * eh.x + ry * eh.y - rz * eh.z;
	dst[4] = m_center - rx * eh.x - ry * eh.y + rz * eh.z;
	dst[5] = m_center + rx * eh.x - ry * eh.y + rz * eh.z;
	dst[6] = m_center - rx * eh.x + ry * eh.y + rz * eh.z;
	dst[7] = m_center + rx * eh.x + ry * eh.y + rz * eh.z;
}

const TypedEntity::Type AOrientedBox::type() const
{ return TOrientedBox; }

const BoundingBox AOrientedBox::calculateBBox() const
{
	Vector3F p[8];
	getBoxVertices(p);
	BoundingBox box;
	for(unsigned i = 0; i < 8; i++) {
		box.updateMin(p[i]);
		box.updateMax(p[i]);
	}
	return box;
}

}
//:~
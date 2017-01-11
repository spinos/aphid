/*
 *  AOrientedBox.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <math/AOrientedBox.h>
#include <math/Ray.h>

namespace aphid {

AOrientedBox::AOrientedBox() {}
AOrientedBox::~AOrientedBox() {}

void AOrientedBox::setCenter(const Vector3F & p)
{ m_center = p; }

void AOrientedBox::setOrientation(const Matrix33F & m,
				const Matrix33F::RotateOrder & rod)
{ 
	switch (rod) {
		case Matrix33F::YZX: /// swap ZXY
			m_orientation.setRow(0, m.row(2) );
			m_orientation.setRow(1, m.row(0) );
			m_orientation.setRow(2, m.row(1) );
			break;
		case Matrix33F::ZXY: /// swap YZX
			m_orientation.setRow(0, m.row(1) );
			m_orientation.setRow(1, m.row(2) );
			m_orientation.setRow(2, m.row(0) );
			break;
		case Matrix33F::XZY:
			m_orientation.setRow(0, m.row(0) );
			m_orientation.setRow(1, m.row(2) );
			m_orientation.setRow(2, m.row(1) );
			break;
		case Matrix33F::YXZ:
			m_orientation.setRow(0, m.row(1) );
			m_orientation.setRow(1, m.row(0) );
			m_orientation.setRow(2, m.row(2) );
			break;
		case Matrix33F::ZYX:
			m_orientation.setRow(0, m.row(2) );
			m_orientation.setRow(1, m.row(1) );
			m_orientation.setRow(2, m.row(0) );
			break;
		default:
			m_orientation = m; 
			break;
	}
}

void AOrientedBox::setExtent(const Vector3F & p,				
							const Matrix33F::RotateOrder & rod)
{ 
	switch (rod) {
		case Matrix33F::YZX: /// swap ZXY
			m_extent.set(p.z, p.x, p.y);
			break;
		case Matrix33F::ZXY: /// swap YZX
			m_extent.set(p.y, p.z, p.x);
			break;
		case Matrix33F::XZY:
			m_extent.set(p.x, p.z, p.y);
			break;
		case Matrix33F::YXZ:
			m_extent.set(p.y, p.x, p.z);
			break;
		case Matrix33F::ZYX:
			m_extent.set(p.z, p.y, p.x);
			break;
		default:
			m_extent = p;  
			break;
	}
}

const Vector3F & AOrientedBox::center() const
{ return m_center; }

const Matrix33F & AOrientedBox::orientation() const
{ return m_orientation; }

const Vector3F & AOrientedBox::extent() const
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
	const Vector3F & eh = m_extent;
	dst[0] = m_center - rx * eh.x - ry * eh.y - rz * eh.z;
	dst[1] = m_center + rx * eh.x - ry * eh.y - rz * eh.z;
	dst[2] = m_center - rx * eh.x + ry * eh.y - rz * eh.z;
	dst[3] = m_center + rx * eh.x + ry * eh.y - rz * eh.z;
	dst[4] = m_center - rx * eh.x - ry * eh.y + rz * eh.z;
	dst[5] = m_center + rx * eh.x - ry * eh.y + rz * eh.z;
	dst[6] = m_center - rx * eh.x + ry * eh.y + rz * eh.z;
	dst[7] = m_center + rx * eh.x + ry * eh.y + rz * eh.z;
}

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

void AOrientedBox::set8DOPExtent(const float & x0, const float & x1,
						const float & y0, const float & y1)
{
	m_8DOPExtent[0] = x0;
	m_8DOPExtent[1] = x1;
	m_8DOPExtent[2] = y0;
	m_8DOPExtent[3] = y1;
}

const float * AOrientedBox::dopExtent() const
{ return m_8DOPExtent; }

void AOrientedBox::limitMinThickness(const float & x)
{ if(m_extent.z < x) m_extent.z = x; }

Vector3F AOrientedBox::get8DOPFaceX() const
{
	Vector3F rgt(.7071067f, .7071067f, 0.f);
	return m_orientation.transform(rgt);
}

Vector3F AOrientedBox::get8DOPFaceY() const
{
	Vector3F up(-.7071067f, .7071067f, 0.f);
	return m_orientation.transform(up);
}

std::ostream& operator<<(std::ostream &output, const AOrientedBox & p)
{ 
	output << "(" << p.orientation() << ",\n"  
			<< p.center() << ",\n"
			<< p.extent() << ")"; 
	return output;
}

void AOrientedBox::calculateCenterExtents(const float * p,
					const int & np)
{
	m_center.set(0.f, 0.f, 0.f);
	
	BoundingBox bbox;
	
	for(int i=0;i<np;++i) {
		int j = i * 3;
		bbox.expandBy(Vector3F(p[j], p[j+1], p[j+2]) );
	}
	
	m_center = bbox.center();
	
	m_extent.set(bbox.distance(0) * .5f,
				bbox.distance(1) * .5f,
				bbox.distance(2) * .5f);
	
/// rotate 45 degs
	const Vector3F dx = m_orientation.row(0);
	const Vector3F dy = m_orientation.row(1);
	const Vector3F dz = m_orientation.row(2);
	Vector3F rgt = dx + dy;
	Vector3F up = dz.cross(rgt);
	up.normalize();
	rgt.normalize();
	
	Matrix33F invhalfspace(rgt, up, dz);
	invhalfspace.inverse();
	
	m_8DOPExtent[0] = 1e10f, m_8DOPExtent[1] = -1e10f;
	m_8DOPExtent[2] = 1e10f, m_8DOPExtent[3] = -1e10f;
	
	for(int i=0;i<np;++i) {
		int j = i * 3;
		Vector3F vp(p[j], p[j+1], p[j+2]);
		vp -= m_center;
		vp = invhalfspace.transform(vp);
		
		if(m_8DOPExtent[0] > vp.x ) {
			m_8DOPExtent[0] = vp.x;
		}
		if(m_8DOPExtent[1] < vp.x ) {
			m_8DOPExtent[1] = vp.x;
		}
		if(m_8DOPExtent[2] > vp.y ) {
			m_8DOPExtent[2] = vp.y;
		}
		if(m_8DOPExtent[3] < vp.y ) {
			m_8DOPExtent[3] = vp.y;
		}
	}
	
}

}
//:~
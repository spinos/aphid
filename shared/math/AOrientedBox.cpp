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
					const int & np,
					const BoundingBox * box)
{
	Matrix33F invspace(m_orientation);
	invspace.inverse();

	BoundingBox bbox;
	
	for(int i=0;i<np;++i) {
		int j = i * 3;
		Vector3F vp(p[j], p[j+1], p[j+2]);
		box->putInside(vp);
		vp = invspace.transform(vp);
		bbox.expandBy(vp);
	}
	
	Vector3F offset = bbox.center();
	m_center = m_orientation.transform(offset);
	
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
		box->putInside(vp);
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

void AOrientedBox::caluclateOrientation(const BoundingBox * box)
{
	int axis[3];
	box->getSizeOrder(axis);
	
	Vector3F s(1,0,0);
	if(axis[0] == 1) {
		s.set(0,1,0);
	}
	if(axis[0] == 2) {
		s.set(0,0,1);
	}
	Vector3F f(0,0,1);
	if(axis[2] == 0) {
		f.set(1,0,0);
	}
	if(axis[2] == 1) {
		f.set(0,1,0);
	}
	Vector3F u = f.cross(s);
	m_orientation.fill(s,u,f);
}

void AOrientedBox::calculateCenterExtents(const BoundingBox * box,
						const float * sx)
{
	int axis[3];
	box->getSizeOrder(axis);
	
	Matrix33F invspace(m_orientation);
	invspace.inverse();

	BoundingBox bbox;
	
	for(int i=0;i<8;++i) {
		Vector3F vp = invspace.transform(box->X(i) );
		bbox.expandBy(vp);
	}
	
	Vector3F offset = bbox.center();
	m_center = m_orientation.transform(offset);
	
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
	
	float yox = bbox.distance(1) / bbox.distance(0);
	
	m_8DOPExtent[0] = 1e10f, m_8DOPExtent[1] = -1e10f;
	m_8DOPExtent[2] = 1e10f, m_8DOPExtent[3] = -1e10f;
	
	for(int i=0;i<8;++i) {
		Vector3F vp = box->X(i);
		vp -= m_center;
		
		float * vr = (float * )&vp;
/// move
		if(vr[axis[0]] < 0) {
			if(vr[axis[1]] < 0) {
				vp *= remap(sx[0]);
			} else {
				vp *= remap(sx[1]);
			}
		} else {
			if(vr[axis[1]] < 0) {
				vp *= remap(sx[2]);
			} else {
				vp *= remap(sx[3]);
			}
		}
		
		vp = invhalfspace.transform(vp);

		if(m_8DOPExtent[0] > vr[0] ) {
			m_8DOPExtent[0] = vr[0];
		}
		if(m_8DOPExtent[1] < vr[0] ) {
			m_8DOPExtent[1] = vr[0];
		}
		
		if(m_8DOPExtent[2] > vr[1] ) {
			m_8DOPExtent[2] = vr[1];
		}
		if(m_8DOPExtent[3] < vr[1] ) {
			m_8DOPExtent[3] = vr[1];
		}
	}
}

float AOrientedBox::remap(float x)
{
	if(x>0) {
		return .7 + .1 * x;
	}
	return .7 + .3 * x;
}

}
//:~
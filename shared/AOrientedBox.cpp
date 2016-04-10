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

void AOrientedBox::set8DOPExtent(const float & x0, const float & x1,
						const float & y0, const float & y1)
{
	m_8DOPExtent[0] = x0;
	m_8DOPExtent[1] = x1;
	m_8DOPExtent[2] = y0;
	m_8DOPExtent[3] = y1;
}

void AOrientedBox::get8DOPVertices(Vector3F * dst) const
{
	const Vector3F dx = m_orientation.row(0);
	const Vector3F dy = m_orientation.row(1);
	const Vector3F dz = m_orientation.row(2);
	Vector3F rgt = dx + dy;
	Vector3F up = dz.cross(rgt);
	up.normalize();
	rgt.normalize();
	
	Matrix33F rot(rgt, up, dz);
	
	const Vector3F rx = rot.row(0);
	const Vector3F ry = rot.row(1);
	const Vector3F rz = rot.row(2);
	const float ez = m_extent.z;
	dst[0] = m_center + rx * m_8DOPExtent[0] + ry * m_8DOPExtent[2] - rz * ez;
	dst[1] = m_center + rx * m_8DOPExtent[1] + ry * m_8DOPExtent[2] - rz * ez;
	dst[2] = m_center + rx * m_8DOPExtent[0] + ry * m_8DOPExtent[3] - rz * ez;
	dst[3] = m_center + rx * m_8DOPExtent[1] + ry * m_8DOPExtent[3] - rz * ez;
	dst[4] = m_center + rx * m_8DOPExtent[0] + ry * m_8DOPExtent[2] + rz * ez;
	dst[5] = m_center + rx * m_8DOPExtent[1] + ry * m_8DOPExtent[2] + rz * ez;
	dst[6] = m_center + rx * m_8DOPExtent[0] + ry * m_8DOPExtent[3] + rz * ez;
	dst[7] = m_center + rx * m_8DOPExtent[1] + ry * m_8DOPExtent[3] + rz * ez;
}

void AOrientedBox::get8DOPMesh(Vector3F * vert,
						int * tri,
						int & nvert,
						int & ntri) const
{
/// top side
	nvert = 4;
	vert[0].set(-m_extent.x, -m_extent.y, m_extent.z);
	vert[1].set( m_extent.x, -m_extent.y, m_extent.z);
	vert[2].set( m_extent.x,  m_extent.y, m_extent.z);
	vert[3].set(-m_extent.x,  m_extent.y, m_extent.z);
	ntri = 2;
	tri[0] = 0; tri[1] = 1; tri[2] = 2;
	tri[3] = 0; tri[4] = 2; tri[5] = 3;
/// rotate 45 deg
	Vector3F rgt(1.f, 1.f, 0.f);
	Vector3F up(-1.f, 1.f, 0.f);
	rgt.normalize();
	up.normalize();
	
/// max 8 edges
	int edgei[16];
	int edgec = 4;
	edgei[0] = 0; edgei[1] = 1;
	edgei[2] = 1; edgei[3] = 2;
	edgei[4] = 2; edgei[5] = 3;
	edgei[6] = 3; edgei[7] = 0;
	
	BoundingBox box(-m_extent.x, -m_extent.y, -m_extent.z,
					 m_extent.x,  m_extent.y,  m_extent.z);
	
	Vector3F o = rgt * m_8DOPExtent[0] + up * m_8DOPExtent[3];
	Vector3F d = rgt * m_8DOPExtent[0] + up * m_8DOPExtent[2];
	Vector3F phit;
	Ray incident(o, d);
	float tmin, tmax;
	if(box.intersect(incident, &tmin, &tmax) ) {
		if(tmax - tmin > .2f) {
			phit = incident.travel(tmax);
			vert[0].x = phit.x;
			vert[0].y = phit.y;
			
			phit = incident.travel(tmin);
			vert[nvert].x = phit.x;
			vert[nvert].y = phit.y;
			vert[nvert].z = m_extent.z;
			tri[ntri*3] = 0; tri[ntri*3+1] = nvert-1; tri[ntri*3+2] = nvert;
			edgei[7] = nvert;
			edgei[edgec*2] = nvert;
			edgei[edgec*2+1] = 0;
			edgec++;
			nvert++;
			ntri++;
		}
	}
	
	o = rgt * m_8DOPExtent[0] + up * m_8DOPExtent[2];
	d = rgt * m_8DOPExtent[1] + up * m_8DOPExtent[2];
	incident = Ray(o, d);
	if(box.intersect(incident, &tmin, &tmax) ) {
		if(tmax - tmin > .2f) {
			phit = incident.travel(tmax);
			vert[1].x = phit.x;
			vert[1].y = phit.y;
			
			phit = incident.travel(tmin);
			vert[nvert].x = phit.x;
			vert[nvert].y = phit.y;
			vert[nvert].z = m_extent.z;
			tri[ntri*3] = 0; tri[ntri*3+1] = nvert; tri[ntri*3+2] = 1;
			edgei[1] = nvert;
			edgei[edgec*2] = nvert;
			edgei[edgec*2+1] = 1;
			edgec++;
			nvert++;
			ntri++;
		}
	}
	
	o = rgt * m_8DOPExtent[1] + up * m_8DOPExtent[2];
	d = rgt * m_8DOPExtent[1] + up * m_8DOPExtent[3];
	incident = Ray(o, d);
	if(box.intersect(incident, &tmin, &tmax) ) {
		if(tmax - tmin > .2f) {
			phit = incident.travel(tmax);
			vert[2].x = phit.x;
			vert[2].y = phit.y;
			
			phit = incident.travel(tmin);
			vert[nvert].x = phit.x;
			vert[nvert].y = phit.y;
			vert[nvert].z = m_extent.z;
			tri[ntri*3] = 1; tri[ntri*3+1] = nvert; tri[ntri*3+2] = 2;
			edgei[3] = nvert;
			edgei[edgec*2] = nvert;
			edgei[edgec*2+1] = 2;
			edgec++;
			nvert++;
			ntri++;
		}
	}
	
	o = rgt * m_8DOPExtent[1] + up * m_8DOPExtent[3];
	d = rgt * m_8DOPExtent[0] + up * m_8DOPExtent[3];
	incident = Ray(o, d);
	if(box.intersect(incident, &tmin, &tmax) ) {
		if(tmax - tmin > .2f) {
			phit = incident.travel(tmax);
			vert[3].x = phit.x;
			vert[3].y = phit.y;
			
			phit = incident.travel(tmin);
			vert[nvert].x = phit.x;
			vert[nvert].y = phit.y;
			vert[nvert].z = m_extent.z;
			tri[ntri*3] = 2; tri[ntri*3+1] = nvert; tri[ntri*3+2] = 3;
			edgei[5] = nvert;
			edgei[edgec*2] = nvert;
			edgei[edgec*2+1] = 3;
			edgec++;
			nvert++;
			ntri++;
		}
	}

/// copy to bottom side
	for(int i=0; i<nvert; ++i) {
		vert[i+nvert] = vert[i];
		vert[i+nvert].z = -m_extent.z;
	}
	
	nvert += nvert;
	for(int i=0; i<nvert; ++i) {
		vert[i] = m_orientation.transform(vert[i]) + m_center;
	}
	
	for(int i=0; i<ntri; ++i) {
		tri[(i+ ntri)*3] = tri[i*3] + edgec;
		tri[(i+ ntri)*3+1] = tri[i*3+2] + edgec;
		tri[(i+ ntri)*3+2] = tri[i*3+1] + edgec;
	}
	ntri += ntri;
	
/// connect top and bottom
	for(int i=0; i<edgec; ++i) {
		int i1 = i+1;
		if(i1 == edgec) i1 = 0;
		
		tri[ntri*3] = edgei[i];
		tri[ntri*3+1] = edgei[i] + edgec;
		tri[ntri*3+2] = edgei[i1];
		ntri++;
		tri[ntri*3] = edgei[i] + edgec;
		tri[ntri*3+1] = edgei[i1] + edgec;
		tri[ntri*3+2] = edgei[i1];
		ntri++;
	}
	
	///std::cout<<"\n n edge "<<edgec
		///		<<"\n v "<<nvert
			///	<<"\n t "<<ntri;
	
}

void AOrientedBox::limitMinThickness(const float & x)
{ if(m_extent.z < x) m_extent.z = x; }

}
//:~
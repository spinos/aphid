/*
 *  ParallelTransport.cpp
 *  
 *
 *  Created by jian zhang on 1/8/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "ParallelTransport.h"
#include <math/Matrix33F.h>
#include <math/miscfuncs.h>

namespace aphid {

ParallelTransport::ParallelTransport()
{}

void ParallelTransport::CurvatureBinormal(Vector3F& dst,
			const Vector3F& e0, const Vector3F& e1)
{
	dst = e0.cross(e1) * 2.f / (e0.length() * e1.length() + e0.dot(e1));
}

void ParallelTransport::ExtractSinAndCos(float& sinPhi, float& cosPhi,
			const float& kdk)
{
	cosPhi = sqrt(4.f / (4.f + kdk));
	sinPhi = sqrt(kdk / (4.f + kdk));	
}

void ParallelTransport::Rotate(Vector3F& u0, 
						const Vector3F& e0, const Vector3F& e1)
{
	Vector3F axis;
	CurvatureBinormal(axis, e0, e1);
	float sinPhi, cosPhi;
	float magnitude = axis.dot(axis);
	ExtractSinAndCos(magnitude, sinPhi, cosPhi);
	if(1.f - cosPhi < 1e-6f) {
		//u0 = e1.cross(u0);
		//u0 = u0.cross(e1);
		//u0.normalize();
		return;
	}
	
	Quaternion q(cosPhi, axis.normal() * sinPhi);
	Quaternion p(0.f, u0);
	p = q * p;
	
	const Vector3F oldu = u0;
	u0.set(p.x, p.y, p.z);
	u0.normalize();
	std::cout<<"\n "<<oldu.dot(u0);
	
}

void ParallelTransport::FirstFrame(Matrix33F& frm, 
					const Vector3F& e0, const Vector3F& refv)
{
	Vector3F N = refv;
	Vector3F B = e0.cross(N);
	if(B.length() < 1e-3f) {
		N = refv.perpendicular();
		B = e0.cross(N);
	}
	B.normalize();
	N = B.cross(e0);
	N.normalize();
	frm.fill(e0.normal(), N, B);
}

void ParallelTransport::RotateFrame(Matrix33F& frm, 
					const Vector3F& e0, const Vector3F& e1)
{
	const Vector3F t0 = e0.normal();
	const Vector3F t1 = e1.normal();
	Vector3F t0xt1 = t0.cross(t1);
	if(t0xt1.length2() < 1e-6f)
		return;
		
	const float ang = acos(t0.dot(t1));
	if(ang < 1e-3f)
		return;
/// rotate around binormal		
	Quaternion q(ang, (t0.cross(t1)).normal());
	Matrix33F rot(q);
	frm *= rot;
}

Vector3F ParallelTransport::FrameUp(const Matrix33F& frm)
{
	return Vector3F(frm.M(1,0), frm.M(1,1), frm.M(1,2) );
}

}
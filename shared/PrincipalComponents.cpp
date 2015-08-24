/*
 *  PrincipalComponents.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "PrincipalComponents.h"
#include <BoundingBox.h>

PrincipalComponents::PrincipalComponents() {}
PrincipalComponents::~PrincipalComponents() {}

AOrientedBox PrincipalComponents::analyze(Vector3F * pos, unsigned n)
{
	Vector3F bar = Vector3F::Zero;
	unsigned i=0;
	for(;i<n;i++) bar += pos[i];
	bar *= 1.f/(float)n;
	
	for(i=0;i<n;i++) pos[i] -= bar;
	
// is symmetric
	Matrix33F covarianceMatrix;
	*covarianceMatrix.m(0,0) = covarianceXX(pos, n);
	*covarianceMatrix.m(0,1) = covarianceXY(pos, n);
	*covarianceMatrix.m(0,2) = covarianceXZ(pos, n);
	*covarianceMatrix.m(1,0) = covarianceMatrix.M(0,1);
	*covarianceMatrix.m(1,1) = covarianceYY(pos, n);
	*covarianceMatrix.m(1,2) = covarianceYZ(pos, n);
	*covarianceMatrix.m(2,0) = covarianceMatrix.M(0,2);
	*covarianceMatrix.m(2,1) = covarianceMatrix.M(1,2);
	*covarianceMatrix.m(2,2) = covarianceZZ(pos, n);
	
	//float domegv;
	//std::cout//<<"\n marix "<<covarianceMatrix
	//<<"\n dominant eigen vec "<<covarianceMatrix.eigenVector(domegv);
	//std::cout<<"\n dominent eigen val "<<domegv;
	//std::cout<<"\n eigen vals "<<covarianceMatrix.eigenValues();
	
	Vector3F egv;
	Matrix33F egs = covarianceMatrix.eigenSystem(egv);
	AOrientedBox r;
	r.setOrientation(egs);
	
	Matrix33F invspace(egs);
	invspace.inverse();
	BoundingBox bb;
	for(i=0;i<n;i++) {
		pos[i] = invspace.transform(pos[i]);
		bb.expandBy(pos[i]);
	}
	
	Vector3F offset = bb.center();
	offset = egs.transform(offset);
	r.setCenter(bar + offset);
	
	Vector3F ext(bb.distance(0) * .5f, bb.distance(1) * .5f, bb.distance(2) * .5f);
	r.setExtent(ext);
	return r;
}

float PrincipalComponents::covarianceXX(Vector3F * pos, unsigned n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += pos[i].x * pos[i].x;
	return res / (float)n;
}

float PrincipalComponents::covarianceXY(Vector3F * pos, unsigned n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += pos[i].x * pos[i].y;
	return res / (float)n;
}

float PrincipalComponents::covarianceXZ(Vector3F * pos, unsigned n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += pos[i].x * pos[i].z;
	return res / (float)n;
}

float PrincipalComponents::covarianceYX(Vector3F * pos, unsigned n) const
{ return covarianceXY(pos, n); }

float PrincipalComponents::covarianceYY(Vector3F * pos, unsigned n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += pos[i].y * pos[i].y;
	return res / (float)n;
}

float PrincipalComponents::covarianceYZ(Vector3F * pos, unsigned n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += pos[i].y * pos[i].z;
	return res / (float)n;
}

float PrincipalComponents::covarianceZX(Vector3F * pos, unsigned n) const
{ return covarianceXZ(pos, n); }

float PrincipalComponents::covarianceZY(Vector3F * pos, unsigned n) const
{ return covarianceYZ(pos, n); }

float PrincipalComponents::covarianceZZ(Vector3F * pos, unsigned n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += pos[i].z * pos[i].z;
	return res / (float)n;
}
//:~
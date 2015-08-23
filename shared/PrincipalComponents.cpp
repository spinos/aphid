/*
 *  PrincipalComponents.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "PrincipalComponents.h"

PrincipalComponents::PrincipalComponents() {}
PrincipalComponents::~PrincipalComponents() {}

void PrincipalComponents::analyze(Vector3F * pos, unsigned n)
{
	Vector3F bar = Vector3F::Zero;
	unsigned i=0;
	for(;i<n;i++) bar += pos[i];
	bar *= 1.f/(float)n;
	
	for(i=0;i<n;i++) pos[i] -= bar;
	
	Matrix33F covarianceMatrix;
	*covarianceMatrix.m(0,0) = covarianceXX(pos, n);
	*covarianceMatrix.m(0,1) = covarianceXY(pos, n);
	*covarianceMatrix.m(0,2) = covarianceXZ(pos, n);
	*covarianceMatrix.m(1,0) = covarianceYX(pos, n);
	*covarianceMatrix.m(1,1) = covarianceYY(pos, n);
	*covarianceMatrix.m(1,2) = covarianceYZ(pos, n);
	*covarianceMatrix.m(2,0) = covarianceZX(pos, n);
	*covarianceMatrix.m(2,1) = covarianceZY(pos, n);
	*covarianceMatrix.m(2,2) = covarianceZZ(pos, n);
	
	//float domegv;
	//std::cout//<<"\n marix "<<covarianceMatrix
	//<<"\n dominant eigen vec "<<covarianceMatrix.eigenVector(domegv);
	//std::cout<<"\n dominent eigen val "<<domegv;
	//std::cout<<"\n eigen vals "<<covarianceMatrix.eigenValues();
	
	Vector3F egv;
	std::cout<<"\n eigen sys "<<covarianceMatrix.eigenSystem(egv);
	std::cout<<"\n all eigen vals "<<egv;
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
/*
 *  PrincipalComponents.cpp
 *  
 *
 *  Created by jian zhang on 4/17/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <geom/PrincipalComponents.h>

namespace aphid {

PCASlave::PCASlave() :
m_v(0.f)
{}

const float & PCASlave::result() const
{ return m_v; }

void PCASlave::covarianceXX(const Vector3F * p, int n)
{
	int i=0;
	for(;i<n;i++) m_v += p[i].x * p[i].x;
	m_v /= (float)n;
}

void PCASlave::covarianceXY(const Vector3F * p, int n)
{
	int i=0;
	for(;i<n;i++) m_v += p[i].x * p[i].y;
	m_v /= (float)n;
}

void PCASlave::covarianceXZ(const Vector3F * p, int n)
{
	int i=0;
	for(;i<n;i++) m_v += p[i].x * p[i].z;
	m_v /= (float)n;
}

void PCASlave::covarianceYY(const Vector3F * p, int n)
{
	int i=0;
	for(;i<n;i++) m_v += p[i].y * p[i].y;
	m_v /= (float)n;
}

void PCASlave::covarianceYZ(const Vector3F * p, int n)
{
	int i=0;
	for(;i<n;i++) m_v += p[i].y * p[i].z;
	m_v /= (float)n;
}

void PCASlave::covarianceZZ(const Vector3F * p, int n)
{
	int i=0;
	for(;i<n;i++) m_v += p[i].z * p[i].z;
	m_v /= (float)n;
}

}
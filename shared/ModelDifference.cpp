/*
 *  ModelDifference.cpp
 *  testadenium
 *
 *  Created by jian zhang on 7/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ModelDifference.h"
#include "AGenericMesh.h"
#include <BaseBuffer.h>
#include <iostream>

ModelDifference::ModelDifference(AGenericMesh * target) : m_target(target)
{
	m_p0 = new BaseBuffer;
	m_p0->create(m_target->numPoints() * 12);
}

ModelDifference::~ModelDifference()
{
	delete m_p0;
}

bool ModelDifference::matchTarget(AGenericMesh * object) const
{
	if(object->numPoints() != m_target->numPoints()) {
		std::cout<<" nv not match\n";
		return false;
	}
	if(object->numIndices() != m_target->numIndices()) {
		std::cout<<" topo not match\n";
		return false;
	}
	
	Vector3F * prest = m_target->points();
	Vector3F * p1 = object->points();
	const unsigned n = m_target->numPoints();
	const Vector3F dv0 = p1[0] - prest[0];
	Vector3F dv;
	unsigned i=1;
	for(;i<n;i++) {
		dv = p1[i] - prest[i];
        if((dv - dv0).length2() > 1e-6f) {
			std::cout<<" shape not match\n";
			return false;
		}
	}
	
	return true;
}

Vector3F ModelDifference::computeCenterOf(const AGenericMesh * object) const
{
	Vector3F * p1 = object->points();
	const unsigned n = object->numPoints();
	Vector3F center = Vector3F::Zero;
	unsigned i=0;
	for(;i<n;i++) center += p1[i];
	center *= 1.f/(float)n;
	return center;
}

Vector3F ModelDifference::resetTranslation(const AGenericMesh * object)
{ 
	m_centers.clear();
	Vector3F p = computeCenterOf(object);
	m_centers.push_back(p);
	return p;
}

Vector3F ModelDifference::addTranslation(const AGenericMesh * object)
{
	Vector3F currentCenter = computeCenterOf(object);
	Vector3F dp = currentCenter - m_centers.back();
	m_centers.push_back(currentCenter);
	return dp;
}

void ModelDifference::moveToObjectSpace(AGenericMesh * object) const
{
	const Vector3F center = m_centers.back();
	const unsigned n = object->numPoints();
	Vector3F * p1 = object->points();
	unsigned i=0;
	for(;i<n;i++) p1[i] -= center;
}

void ModelDifference::computeVelocities(Vector3F * dst, AGenericMesh * object, float oneOverDt)
{
	const unsigned n = object->numPoints();
	Vector3F * p1 = object->points();
	Vector3F * p0 = (Vector3F *)m_p0->data();
	Vector3F q;
	unsigned i=0;
	for(;i<n;i++) {
		q = p1[i];
		dst[i] = q - p0[i];
		dst[i] *= oneOverDt;
		p0[i] = q;
	}
}

const unsigned ModelDifference::numTranslations() const
{ return m_centers.size(); }

const Vector3F ModelDifference::getTranslation(unsigned idx) const
{ return m_centers[idx]; }

const Vector3F ModelDifference::lastTranslation() const
{ return m_centers.back(); }
//:~
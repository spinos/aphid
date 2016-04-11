/*
 *  PrincipalComponents.h
 *  aphid
 *
 *  Created by jian zhang on 8/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <AllMath.h>
#include <AOrientedBox.h>

namespace aphid {

template<typename T>
class PrincipalComponents {
	
	Vector3F * m_pos;
	int m_constrain;
	
public:
	PrincipalComponents();
	virtual ~PrincipalComponents();
	
	void setOrientConstrain(int x);
	AOrientedBox analyze(const T & pos, int n);
	
protected:
	float covarianceXX(int n) const;
	float covarianceXY(int n) const;
	float covarianceXZ(int n) const;
	float covarianceYY(int n) const;
	float covarianceYZ(int n) const;
	float covarianceZZ(int n) const;
	Matrix33F getOrientation(int n);
	
private:
	
};

template<typename T>
PrincipalComponents<T>::PrincipalComponents() :
m_pos(NULL),
m_constrain(0)
{}

template<typename T>
PrincipalComponents<T>::~PrincipalComponents()
{
	if(m_pos) delete[] m_pos;
}

template<typename T>
void PrincipalComponents<T>::setOrientConstrain(int x)
{ m_constrain = x; }

template<typename T>
Matrix33F PrincipalComponents<T>::getOrientation(int n)
{
	if(m_constrain>0) {
		return Matrix33F(Vector3F(1.f, 0.f, 0.f), 
						Vector3F(0.f, 0.f, -1.f),
						Vector3F(0.f, 1.f, 0.f));
	}
	
// is symmetric
	Matrix33F covarianceMatrix;
	*covarianceMatrix.m(0,0) = covarianceXX(n);
	*covarianceMatrix.m(0,1) = covarianceXY(n);
	*covarianceMatrix.m(0,2) = covarianceXZ(n);
	*covarianceMatrix.m(1,0) = covarianceMatrix.M(0,1);
	*covarianceMatrix.m(1,1) = covarianceYY(n);
	*covarianceMatrix.m(1,2) = covarianceYZ(n);
	*covarianceMatrix.m(2,0) = covarianceMatrix.M(0,2);
	*covarianceMatrix.m(2,1) = covarianceMatrix.M(1,2);
	*covarianceMatrix.m(2,2) = covarianceZZ(n);
	
	//float domegv;
	//std::cout//<<"\n marix "<<covarianceMatrix
	//<<"\n dominant eigen vec "<<covarianceMatrix.eigenVector(domegv);
	//std::cout<<"\n dominent eigen val "<<domegv;
	//std::cout<<"\n eigen vals "<<covarianceMatrix.eigenValues();
	Vector3F egv;
	return covarianceMatrix.eigenSystem(egv);
}

template<typename T>
AOrientedBox PrincipalComponents<T>::analyze(const T & pos, int n)
{
	Vector3F bar = Vector3F::Zero;
	int i=0;
	for(;i<n;i++) bar += pos.at(i);
	bar *= 1.f/(float)n;
	
	m_pos = new Vector3F[n];
	for(i=0;i<n;i++) m_pos[i] = pos.at(i) - bar;
	
	Matrix33F egs = getOrientation(n);
	AOrientedBox r;
	r.setOrientation(egs);
	
/// extent in local space
	Matrix33F invspace(egs);
	invspace.inverse();
	BoundingBox bb;
	for(i=0;i<n;i++) {
		bb.expandBy(invspace.transform(m_pos[i]) );
	}
	
	Vector3F offset = bb.center();
	offset = egs.transform(offset);
	r.setCenter(bar + offset);
	
	Vector3F ext(bb.distance(0) * .5f, bb.distance(1) * .5f, bb.distance(2) * .5f);
	r.setExtent(ext);
	
/// rotate 45 degs
	const Vector3F dx = egs.row(0);
	const Vector3F dy = egs.row(1);
	const Vector3F dz = egs.row(2);
	Vector3F rgt = dx + dy;
	Vector3F up = dz.cross(rgt);
	up.normalize();
	rgt.normalize();
	
	Matrix33F invhalfspace(rgt, up, dz);
	invhalfspace.inverse();
	
	float minX = 1e10f, maxX = -1e10f;
	float minY = 1e10f, maxY = -1e10f;
	for(i=0;i<n;i++) {
		Vector3F loc = invhalfspace.transform(m_pos[i] - offset );
		
		if(minX > loc.x ) minX = loc.x;
		if(maxX < loc.x ) maxX = loc.x;
		
		if(minY > loc.y ) minY = loc.y;
		if(maxY < loc.y ) maxY = loc.y;
	}
	
	r.set8DOPExtent(minX, maxX, 
					minY, maxY);
	return r;
}

template<typename T>
float PrincipalComponents<T>::covarianceXX(int n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += m_pos[i].x * m_pos[i].x;
	return res / (float)n;
}

template<typename T>
float PrincipalComponents<T>::covarianceXY(int n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += m_pos[i].x * m_pos[i].y;
	return res / (float)n;
}

template<typename T>
float PrincipalComponents<T>::covarianceXZ(int n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += m_pos[i].x * m_pos[i].z;
	return res / (float)n;
}

template<typename T>
float PrincipalComponents<T>::covarianceYY(int n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += m_pos[i].y * m_pos[i].y;
	return res / (float)n;
}

template<typename T>
float PrincipalComponents<T>::covarianceYZ(int n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += m_pos[i].y * m_pos[i].z;
	return res / (float)n;
}

template<typename T>
float PrincipalComponents<T>::covarianceZZ(int n) const
{
	float res = 0.f;
	unsigned i=0;
	for(;i<n;i++) res += m_pos[i].z * m_pos[i].z;
	return res / (float)n;
}

}
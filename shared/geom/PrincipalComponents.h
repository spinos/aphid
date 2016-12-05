/*
 *  PrincipalComponents.h
 *  aphid
 *
 *  Created by jian zhang on 8/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APHID_VOXEL_PRINCIPALCOMPONENTS_H
#define APHID_VOXEL_PRINCIPALCOMPONENTS_H
#include <AllMath.h>
#include <AOrientedBox.h>
#include <boost/thread.hpp>  

namespace aphid {

class PCASlave {
	
	float m_v;
	
public:
	PCASlave();
	
	void covarianceXX(const Vector3F * p, int n);
	void covarianceXY(const Vector3F * p, int n);
	void covarianceXZ(const Vector3F * p, int n);
	void covarianceYY(const Vector3F * p, int n);
	void covarianceYZ(const Vector3F * p, int n);
	void covarianceZZ(const Vector3F * p, int n);
	
	const float & result() const;
	
};


template<typename T>
class PrincipalComponents {
	
	Vector3F * m_pos;
	int m_constrain;
	
public:
	PrincipalComponents();
	virtual ~PrincipalComponents();
	
	void setOrientConstrain(int x);
/// find orientaion and bbox
	AOrientedBox analyze(const T & pos, int n,
					const Matrix33F::RotateOrder & rod = Matrix33F::XYZ);
	
protected:
	Matrix33F getOrientation(int n) const;
	
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
Matrix33F PrincipalComponents<T>::getOrientation(int n) const
{
	if(m_constrain>0) {
		return Matrix33F(Vector3F(1.f, 0.f, 0.f), 
						Vector3F(0.f, 0.f, -1.f),
						Vector3F(0.f, 1.f, 0.f));
	}
	
	PCASlave slave[6];
	boost::thread workTr[6];
	workTr[0] = boost::thread(boost::bind(&PCASlave::covarianceXX, 
										&slave[0], 
										m_pos, n) );
	workTr[1] = boost::thread(boost::bind(&PCASlave::covarianceXY, 
										&slave[1], 
										m_pos, n) );
	workTr[2] = boost::thread(boost::bind(&PCASlave::covarianceXZ, 
										&slave[2], 
										m_pos, n) );
	workTr[3] = boost::thread(boost::bind(&PCASlave::covarianceYY, 
										&slave[3], 
										m_pos, n) );
	workTr[4] = boost::thread(boost::bind(&PCASlave::covarianceYZ, 
										&slave[4], 
										m_pos, n) );
	workTr[5] = boost::thread(boost::bind(&PCASlave::covarianceZZ, 
										&slave[5], 
										m_pos, n) );
										
	for(int i=0; i<6; ++i) workTr[i].join();
	
	Matrix33F covarianceMatrix;
	*covarianceMatrix.m(0,0) = slave[0].result();
	*covarianceMatrix.m(0,1) = slave[1].result();
	*covarianceMatrix.m(0,2) = slave[2].result();
	*covarianceMatrix.m(1,0) = covarianceMatrix.M(0,1);
	*covarianceMatrix.m(1,1) = slave[3].result();
	*covarianceMatrix.m(1,2) = slave[4].result();
	*covarianceMatrix.m(2,0) = covarianceMatrix.M(0,2);
	*covarianceMatrix.m(2,1) = covarianceMatrix.M(1,2);
	*covarianceMatrix.m(2,2) = slave[5].result();

	Vector3F egv;
	return covarianceMatrix.eigenSystem(egv);
}

template<typename T>
AOrientedBox PrincipalComponents<T>::analyze(const T & pos, int n,
									const Matrix33F::RotateOrder & rod)
{
	Vector3F bar = Vector3F::Zero;
	int i=0;
	for(;i<n;i++) bar += pos.at(i);
	bar *= 1.f/(float)n;
	
	m_pos = new Vector3F[n];
	for(i=0;i<n;i++) m_pos[i] = pos.at(i) - bar;
	
	Matrix33F egs = getOrientation(n);
	AOrientedBox r;
/// order of components
	r.setOrientation(egs, rod);
	
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
	r.setExtent(ext, rod);
	
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

}
#endif
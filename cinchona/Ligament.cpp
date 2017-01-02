/*
 *  Ligament.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Ligament.h"
#include <math/HermiteInterpolatePiecewise.h>
#include <AllMath.h>

using namespace aphid;

Ligament::Ligament(const int & np)
{
	m_interp = new HermiteInterpolatePiecewise<float, Vector3F >(np);
	for(int i=0;i<np;++i) {
		m_interp->setPieceBegin(i, Vector3F(i,0,0), Vector3F(1,0,0) );
		m_interp->setPieceEnd(i, Vector3F(i+1,0,0), Vector3F(1,0,0) );
	}
	
	m_knotPoint = new Vector3F[np+1]; 
	m_knotOffset = new Vector3F[np+1];
	m_knotTangent = new Vector3F[np+1];
	for(int i=0;i<=np;++i) {
		m_knotPoint[i].set(i, 0.f, 0.f);
		m_knotOffset[i].set(0.f, 0.f, 0.f);
		m_knotTangent[i].set(1.f, 0.f, 0.f);
	}
	
}

Ligament::~Ligament()
{
	delete m_interp;
	delete[] m_knotPoint;
	delete[] m_knotOffset;
	delete[] m_knotTangent;
}

void Ligament::setKnotOffset(const int & idx,
				const aphid::Vector3F & v)
{
	m_knotOffset[idx] = v;
}

void Ligament::setKnotPoint(const int & idx,
				const aphid::Vector3F & v)
{
	m_knotPoint[idx] = v;
}

void Ligament::setKnotTangent(const int & idx,
				const aphid::Vector3F & v)
{
	m_knotTangent[idx] = v;
}

void Ligament::update()
{
	const int n = m_interp->numPieces() + 1;
	for(int i=0;i<n;++i) {
		setKnot(i, m_knotPoint[i] + m_knotOffset[i], 
				m_knotTangent[i]);
	}
}

void Ligament::setKnot(const int & idx,
				const aphid::Vector3F & pt,
				const aphid::Vector3F & tg)
{
	if(idx==0) {
		m_interp->setPieceBegin(0, pt, tg);
	} else if (idx == m_interp->numPieces() ) {
		m_interp->setPieceEnd(idx - 1, pt, tg);
	} else {
		m_interp->setPieceEnd(idx - 1, pt, tg);
		m_interp->setPieceBegin(idx, pt, tg);
	}
}

Vector3F Ligament::getPoint(const int & idx,
				const float & param) const
{
	return m_interp->interpolate(idx, param);
}

const int & Ligament::numPieces() const
{ return m_interp->numPieces(); }

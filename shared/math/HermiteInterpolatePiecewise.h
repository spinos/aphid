/*
 *  HermiteInterpolatePiecewise.h
 *  
 *
 *  Created by jian zhang on 1/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MATH_HERMITE_INTERPOLATE_PIECE_WISE_H
#define APH_MATH_HERMITE_INTERPOLATE_PIECE_WISE_H

#include "linspace.h"

namespace aphid {

template<typename T1, typename T2>
class HermiteInterpolatePiecewise {

	T2 * m_pt;
	T2 * m_tg;
	int m_numPieces;
	
public:
	HermiteInterpolatePiecewise(const int & np = 3);
	virtual ~HermiteInterpolatePiecewise();
	
/// create n pieces
	void create(const int & npieces);
/// idx-th piece
	void setPieceBegin(const int & idx,
				const T2 & p, const T2 & t);
	void setPieceEnd(const int & idx,
				const T2 & p, const T2 & t);
/// x [0,1] within idx-th piece
	T2 interpolate(const int & idx,
				const T1 & x) const;
/// tangent	x [0,1] within idx-th piece			
	T2 derivative(const int & idx,
				const T1 & x) const;
				
	const int & numPieces() const;
	
/// length of each piece and sum
	void getLengths(float * & segmentLs,
					int & np,
					float & totalL) const;
/// access
	const T2& Pnt(const int& i) const;
	const T2& Tng(const int& i) const;
	
protected:

private:
};

template<typename T1, typename T2>
HermiteInterpolatePiecewise<T1, T2>::HermiteInterpolatePiecewise(const int & np) :
m_pt(0),
m_tg(0)
{ create(np); }

template<typename T1, typename T2>
HermiteInterpolatePiecewise<T1, T2>::~HermiteInterpolatePiecewise()
{
	if(m_pt) {
		delete[] m_pt;
	}
	if(m_tg) {
		delete[] m_tg;
	}
}

template<typename T1, typename T2>
void HermiteInterpolatePiecewise<T1, T2>::create(const int & npieces)
{
	m_numPieces = npieces;
	if(m_pt) {
		delete[] m_pt;
	}
	if(m_tg) {
		delete[] m_tg;
	}
/// begin and end per seg
	m_pt = new T2[npieces << 1];
	m_tg = new T2[npieces << 1];
}

template<typename T1, typename T2>
void HermiteInterpolatePiecewise<T1, T2>::setPieceBegin(const int & idx,
				const T2 & p, const T2 & t)
{
	int i = idx<<1;
	m_pt[i] = p;
	m_tg[i] = t;
}

template<typename T1, typename T2>
void HermiteInterpolatePiecewise<T1, T2>::setPieceEnd(const int & idx,
				const T2 & p, const T2 & t)
{
	int i = (idx<<1)+1;
	m_pt[i] = p;
	m_tg[i] = t;
}

template<typename T1, typename T2>
T2 HermiteInterpolatePiecewise<T1, T2>::interpolate(const int & idx,
				const T1 & x) const
{
	int i = idx<<1;
	T1 s2 = x * x;
	T1 s3 = s2 * x;
	T1 h1 =  2.f * s3 - 3.f * s2 + 1.f;          // calculate basis function 1
	T1 h2 = -2.f * s3 + 3.f * s2;              // calculate basis function 2
	T1 h3 =   s3 - 2.f * s2 + x;         // calculate basis function 3
	T1 h4 =   s3 -  s2;              // calculate basis function 4
	return (m_pt[i] * h1 +                    // multiply and sum all funtions
             m_pt[i+1] * h2 +                    // together to build the interpolated
             m_tg[i] * h3 +                    // point along the curve.
             m_tg[i+1] * h4);
			 
}

template<typename T1, typename T2>
T2 HermiteInterpolatePiecewise<T1, T2>::derivative(const int & idx,
				const T1 & x) const
{
	float x0 = x;
	float x1 = x + 0.01;
	if(x1 > 1.0) {
		x0 = x - 0.01;
		x1 = x;
	}
	
	T2 v0 = interpolate(idx, x0);
	T2 v1 = interpolate(idx, x1);
	
	return (v1 - v0) * 100.0;
}

template<typename T1, typename T2>
const int & HermiteInterpolatePiecewise<T1, T2>::numPieces() const
{ return m_numPieces; }

template<typename T1, typename T2>
void HermiteInterpolatePiecewise<T1, T2>::getLengths(float * & segmentLs,
					int & np,
					float & totalL) const
{
	np = numPieces();
	segmentLs = new float[np];
	totalL = 0.f;
	T1 xs[17];
	linspace<T1>(xs, 0.f, 1.f, 17);
	
	T2 p, q;
	for(int i=0;i<np;++i) {
		segmentLs[i] = 0.f;
		
		p = interpolate(i, xs[0]);
		for(int j=1;j<17;++j) {
			q = interpolate(i, xs[j]);
			
			segmentLs[i] += p.distanceTo(q);
			
			p = q;
		}
		
		totalL += segmentLs[i];
	}
}

template<typename T1, typename T2>
const T2& HermiteInterpolatePiecewise<T1, T2>::Pnt(const int& i) const
{ return m_pt[i]; }

template<typename T1, typename T2>
const T2& HermiteInterpolatePiecewise<T1, T2>::Tng(const int& i) const
{ return m_tg[i]; }

}
#endif
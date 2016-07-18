/*
 *  RedBlueRefine.cpp
 *  
 *
 *  Created by jian zhang on 7/16/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "RedBlueRefine.h"
#include <tetrahedron_math.h>

using namespace aphid;

namespace ttg {

RedBlueRefine::RedBlueRefine()
{}
	
void RedBlueRefine::set(int a, int b, int c, int d)
{
	m_a = a; m_b = b;
	m_c = c; m_d = d;
	setTetrahedronVertices(m_tet[0], a, b, c, d);
	m_N = 1;
/// reset
	m_red[0] = m_red[1] = -1;
	m_blue[0] = m_blue[1] = m_blue[2] = m_blue[3] = -1;
}

void RedBlueRefine::setP(const aphid::Vector3F & a,
			const aphid::Vector3F & b,
			const aphid::Vector3F & c,
			const aphid::Vector3F & d)
{
	m_p[0] = a; m_p[1] = b;
	m_p[2] = c; m_p[3] = d;
}

void RedBlueRefine::evaluateDistance(float a, float b, float c, float d)
{
	int cred = 0, cblue = 0;
	if(splitCondition(a, b) ) {
		m_red[0] = 0;
		cred++;
	}
		
	if(splitCondition(c, d) ) {
		m_red[1] = 0;
		cred++;
	}
		
	if(splitCondition(a, c) ) {
		m_blue[0] = 0;
		cblue++;
	}
		
	if(splitCondition(a, d) ) {
		m_blue[1] = 0;
		cblue++;
	}
		
	if(splitCondition(b, c) ) {
		m_blue[2] = 0;
		cblue++;
	}
		
	if(splitCondition(b, d) ) {
		m_blue[3] = 0;
		cblue++;
	}
		
	m_opt = SpNone;
	
	if(cred == 0) {
		if(cblue == 1)
			m_opt = SpOneBlue;
		else if(cblue == 2) {
			if((m_blue[0] > -1 && m_blue[1] > -1)
				|| (m_blue[2] > -1 && m_blue[3] > -1) )
				m_opt = SpTwoBlueUp;
			else
				m_opt = SpTwoBlueDown;
		}
		else if(cblue == 4)
			m_opt = SpFourBlue;
	}
	else if(cred == 1) {
		if(cblue == 0)
			m_opt = SpOneRed;
		else if(cblue == 1)
			m_opt = SpOneRedOneBlue;
		else if(cblue == 2) {
			if(m_red[1] > -1) m_opt = SpOneRedUpTwoBlue;
			else m_opt = SpOneRedDownTwoBlue;
		}
	}
	else {
		if(cblue == 2)
			m_opt = SpTwoRedTwoBlue;
	}
	
	auxiliarySplit();
	
}

void RedBlueRefine::verbose() const
{
	std::string strOpt("none");
	switch (m_opt) {
		case SpOneRed:
			strOpt = "1 red";
			break;
		case SpOneRedOneBlue:
			strOpt = "1 red 1 blue";
			break;
		case SpOneRedUpTwoBlue:
			strOpt = "1 red up 2 blue";
			break;
		case SpOneRedDownTwoBlue:
			strOpt = "1 red down 2 blue";
			break;
		case SpTwoRedTwoBlue:
			strOpt = "2 red 2 blue";
			break;
		case SpOneBlue:
			strOpt = "1 blue";
			break;
		case SpTwoBlueUp:
			strOpt = "2 blue up";
			break;
		case SpTwoBlueDown:
			strOpt = "2 blue down";
			break;
		case SpFourBlue:
			strOpt = "4 blue";
			break;
		default:
			break;
	}
	std::cout<<"\n split option "<<strOpt;
	
	std::cout<<"\n red "<<m_red[0]<<", "<<m_red[1]
			<<"\n blue "<<m_blue[0]<<", "<<m_blue[1]<<", "<<m_blue[2]<<", "<<m_blue[3];
}

bool RedBlueRefine::hasOption() const
{ return m_opt != SpNone; }

void RedBlueRefine::auxiliarySplit()
{
	if(m_opt == SpOneRedUpTwoBlue) {
/// other red
		m_red[0] = 0;
	}
	else if(m_opt == SpOneRedDownTwoBlue) {
/// other red
		m_red[1] = 0;
	}
	else if(m_opt == SpTwoBlueUp) {
/// red up
		m_red[1] = 0;
	}
	else if(m_opt == SpTwoBlueDown) {
/// red bottom
		m_red[0] = 0;
	}
	else if(m_opt == SpFourBlue) {
/// both red
		m_red[0] = 0;
		m_red[1] = 0;
	}
	else if(m_opt == SpTwoRedTwoBlue) {
/// all blue
		m_blue[0] = 0;
		m_blue[1] = 0;
		m_blue[2] = 0;
		m_blue[3] = 0;
	}	
}

const int & RedBlueRefine::numTetra() const
{ return m_N; }

const ITetrahedron * RedBlueRefine::tetra(int i) const
{ return &m_tet[i]; }

void RedBlueRefine::splitRedEdge(int i, int v)
{ m_red[i] = v; }

void RedBlueRefine::splitBlueEdge(int i, int v)
{ m_blue[i] = v; }

void RedBlueRefine::splitRedEdge(int i, int v, const aphid::Vector3F & p)
{ m_red[i] = v; m_p[4 + i] = p; }

void RedBlueRefine::splitBlueEdge(int i, int v, const aphid::Vector3F & p)
{ m_blue[i] = v; m_p[6 + i] = p; }

bool RedBlueRefine::needSplitRedEdge(int i)
{ return m_red[i] > -1; }

bool RedBlueRefine::needSplitBlueEdge(int i)
{ return m_blue[i] > -1; }

bool RedBlueRefine::splitCondition(const float & a,
					const float & b) const
{ return (a * b) < 0.f; }

Vector3F RedBlueRefine::splitPos(float a, float b,
						const Vector3F & pa,
						const Vector3F & pb) const
{
	if(!splitCondition(a, b) )
		return (pa + pb) * .5f;
		
	float sa = Absolute<float>(a);
	float sb = Absolute<float>(b);
	return pa * (sb / (sa + sb)) + pb * (sa / (sa + sb));
}

int RedBlueRefine::pInd(int i) const
{
	if(i==m_a) return 0;
	if(i==m_b) return 1;
	if(i==m_c) return 2;
	if(i==m_d) return 3;
	if(i==m_red[0]) return 4;
	if(i==m_red[1]) return 5;
	if(i==m_blue[0]) return 6;
	if(i==m_blue[1]) return 7;
	if(i==m_blue[2]) return 8;
	return 9;
}

bool RedBlueRefine::checkTetraVolume(const Vector3F * p) const
{
	float mnvol = 1e20f, mxvol = -1e20f, vol;
	Vector3F tp[4];
	const int n = numTetra();
	int i = 0;
	for(;i<n;++i) {
		const ITetrahedron * t = tetra(i);
		
		tp[0] = p[t->iv0];
		tp[1] = p[t->iv1];
		tp[2] = p[t->iv2];
		tp[3] = p[t->iv3];
		
		vol = tetrahedronVolume(p);
		if(mnvol > vol)
			mnvol = vol;
		if(mxvol < vol)
			mxvol = vol;
			
	}

	std::cout<<"\n min/max tetrahedron volume: "<<mnvol<<" / "<<mxvol;
	if(mnvol < 1e-5f) {
		std::cout<<"\n [ERROR] negative(zero) volume "<<mnvol;
		return false;
	}
	
	return true;
}

bool RedBlueRefine::checkTetraVolume() const
{
	float mnvol = 1e20f, mxvol = -1e20f, vol;
	Vector3F p[4];
	const int n = numTetra();
	int i = 0;
	for(;i<n;++i) {
		const ITetrahedron * t = tetra(i);
		
		p[0] = m_p[pInd(t->iv0)];
		p[1] = m_p[pInd(t->iv1)];
		p[2] = m_p[pInd(t->iv2)];
		p[3] = m_p[pInd(t->iv3)];
		
		vol = tetrahedronVolume(p);
		if(mnvol > vol)
			mnvol = vol;
		if(mxvol < vol)
			mxvol = vol;
			
	}

	// std::cout<<"\n min/max tetrahedron volume: "<<mnvol<<" / "<<mxvol;
	if(mnvol < 1e-5f) {
		std::cout<<"\n [ERROR] negative(zero) volume "<<mnvol;
		return false;
	}
	
	return true;
}

void RedBlueRefine::refine()
{
	if(m_opt == SpNone) return;
	
	switch (m_opt) {
		case SpOneRed:
			oneRedRefine();
			break;
		case SpOneBlue:
			oneBlueRefine();
			break;
		case SpOneRedOneBlue:
			oneRedOneBlueRefine();
			break;
		case SpOneRedUpTwoBlue:
			oneRedUpTwoBlueRefine();
			break;
		case SpOneRedDownTwoBlue:
			oneRedDownTwoBlueRefine();
			break;
		case SpTwoRedTwoBlue:
			twoRedTwoBlueRefine();
			break;
		case SpFourBlue:
			fourBlueRefine();
			break;
		case SpTwoBlueUp:
			twoBlueUpRefine();
			break;
		case SpTwoBlueDown:
			twoBlueDownRefine();
			break;	
		default:
			break;
	}
}
	
void RedBlueRefine::oneRedRefine()
{
	if(m_red[0] > 0) {
		splitRed(0, m_tet[0], m_tet[1], m_red[0]);
	}
	else {
		splitRed(1, m_tet[0], m_tet[1], m_red[1]);
	}
	m_N = 2;
}

void RedBlueRefine::oneBlueRefine()
{
	if(m_blue[0] > 0) {
		splitBlue(0, m_tet[0], m_tet[1], m_blue[0]);
	
	}
	else if(m_blue[1] > 0) {
		splitBlue(1, m_tet[0], m_tet[1], m_blue[1]);
		
	}
	else if(m_blue[2] > 0) {
		splitBlue(2, m_tet[0], m_tet[1], m_blue[2]);
		
	}
	else {
		splitBlue(3, m_tet[0], m_tet[1], m_blue[3]);

	}
	m_N = 2;
}

void RedBlueRefine::splitRed(int i, ITetrahedron & t0, ITetrahedron & t1,
								int v)
{
	int a = t0.iv0;
	int b = t0.iv1;
	int c = t0.iv2;
	int d = t0.iv3;
	if(i==0) {
		setTetrahedronVertices(t0, a, v, c, d);
		setTetrahedronVertices(t1, v, b, c, d);
	}
	else {
		setTetrahedronVertices(t0, a, b, c, v);
		setTetrahedronVertices(t1, a, b, v, d);
	}
}

void RedBlueRefine::splitBlue(int i, ITetrahedron & t0, ITetrahedron & t1,
								int v)
{
	int a = t0.iv0;
	int b = t0.iv1;
	int c = t0.iv2;
	int d = t0.iv3;
	if(i==0) {
		setTetrahedronVertices(t0, a, b, v, d);
		setTetrahedronVertices(t1, v, b, c, d);
	}
	else if(i==1) {
		setTetrahedronVertices(t0, a, b, c, v);
		setTetrahedronVertices(t1, v, b, c, d);
	}
	else if(i==2) {
		setTetrahedronVertices(t0, a, v, c, d);
		setTetrahedronVertices(t1, a, b, v, d);
	}
	else {
		setTetrahedronVertices(t0, a, v, c, d);
		setTetrahedronVertices(t1, a, b, c, v);
	}
}

void RedBlueRefine::oneRedOneBlueRefine()
{
	if(m_red[0] > 0) {
		splitRed(0, m_tet[0], m_tet[1], m_red[0]);
		
		if(m_blue[0] > 0) {
			splitBlue(0, m_tet[0], m_tet[2], m_blue[0]);
	
		}
		else if(m_blue[1] > 0) {
			splitBlue(1, m_tet[0], m_tet[2], m_blue[1]);
			
		}
		else if(m_blue[2] > 0) {
			splitBlue(2, m_tet[1], m_tet[2], m_blue[2]);
			
		}
		else {
			splitBlue(3, m_tet[1], m_tet[2], m_blue[3]);

		}
	}
	else {
		splitRed(1, m_tet[0], m_tet[1], m_red[1]);
		
		if(m_blue[0] > 0) {
			splitBlue(0, m_tet[0], m_tet[2], m_blue[0]);
	
		}
		else if(m_blue[1] > 0) {
			splitBlue(1, m_tet[1], m_tet[2], m_blue[1]);
			
		}
		else if(m_blue[2] > 0) {
			splitBlue(2, m_tet[0], m_tet[2], m_blue[2]);
			
		}
		else {
			splitBlue(3, m_tet[1], m_tet[2], m_blue[3]);

		}
	}
	m_N = 3;
}

void RedBlueRefine::oneRedUpTwoBlueRefine()
{
	setTetrahedronVertices(m_tet[0], m_a, m_red[0], m_c, m_red[1]);
	setTetrahedronVertices(m_tet[1], m_a, m_red[0], m_red[1], m_d);
	setTetrahedronVertices(m_tet[2], m_red[0], m_b, m_c, m_red[1]);
	setTetrahedronVertices(m_tet[3], m_red[0], m_b, m_red[1], m_d);
	
	bool lhs = true;
	if(m_blue[0] > 0) {
		m_tet[0].iv2 = m_blue[0];
	}
	if(m_blue[1] > 0) {
		m_tet[1].iv3 = m_blue[1];
		lhs = false;
	}
	if(m_blue[2] > 0) {
		m_tet[2].iv2 = m_blue[2];
	}
	if(m_blue[3] > 0) {
		m_tet[3].iv3 = m_blue[3];
	}
	
	if(lhs) {
		setTetrahedronVertices(m_tet[4], m_red[0], m_red[1], m_blue[2], m_blue[0]);
		setTetrahedronVertices(m_tet[5], m_red[1], m_c, m_blue[2], m_blue[0]);
	}
	else {
		setTetrahedronVertices(m_tet[4], m_red[0], m_red[1], m_blue[1], m_blue[3]);
		setTetrahedronVertices(m_tet[5], m_red[1], m_d, m_blue[1], m_blue[3]);
	}
	
	m_N = 6;
}

void RedBlueRefine::oneRedDownTwoBlueRefine()
{
	setTetrahedronVertices(m_tet[0], m_a, m_red[0], m_c, m_red[1]);
	setTetrahedronVertices(m_tet[1], m_a, m_red[0], m_red[1], m_d);
	setTetrahedronVertices(m_tet[2], m_red[0], m_b, m_c, m_red[1]);
	setTetrahedronVertices(m_tet[3], m_red[0], m_b, m_red[1], m_d);
	
	bool lhs = true;
	if(m_blue[0] > 0) {
		m_tet[0].iv0 = m_blue[0];
	}
	if(m_blue[1] > 0) {
		m_tet[1].iv0 = m_blue[1];
	}
	if(m_blue[2] > 0) {
		m_tet[2].iv1 = m_blue[2];
		lhs = false;
	}
	if(m_blue[3] > 0) {
		m_tet[3].iv1 = m_blue[3];
	}
	
	if(lhs) {
		setTetrahedronVertices(m_tet[4], m_red[0], m_red[1], m_blue[0], m_blue[1]);
		setTetrahedronVertices(m_tet[5], m_a, m_red[0], m_blue[0], m_blue[1]);	
	}
	else {
		setTetrahedronVertices(m_tet[4], m_red[0], m_red[1], m_blue[3], m_blue[2]);
		setTetrahedronVertices(m_tet[5], m_red[0], m_b, m_blue[2], m_blue[3]);	
	}

	m_N = 6;
}

void RedBlueRefine::twoRedTwoBlueRefine()
{
	setTetrahedronVertices(m_tet[0], m_a, m_red[0], m_blue[0], m_blue[1]);
	setTetrahedronVertices(m_tet[1], m_red[0], m_b, m_blue[2], m_blue[3]);
	setTetrahedronVertices(m_tet[2], m_c, m_red[1], m_blue[0], m_blue[2]);
	setTetrahedronVertices(m_tet[3], m_d, m_red[1], m_blue[3], m_blue[1]);
	
	setTetrahedronVertices(m_tet[4], m_red[0], m_red[1], m_blue[0], m_blue[1]);
	setTetrahedronVertices(m_tet[5], m_red[0], m_red[1], m_blue[1], m_blue[3]);
	setTetrahedronVertices(m_tet[6], m_red[0], m_red[1], m_blue[3], m_blue[2]);
	setTetrahedronVertices(m_tet[7], m_red[0], m_red[1], m_blue[2], m_blue[0]);
	
	m_N = 8;
}

void RedBlueRefine::fourBlueRefine()
{
	setTetrahedronVertices(m_tet[0], m_a, m_red[0], m_blue[0], m_blue[1]);
	setTetrahedronVertices(m_tet[1], m_red[0], m_b, m_blue[2], m_blue[3]);
	setTetrahedronVertices(m_tet[2], m_c, m_red[1], m_blue[0], m_blue[2]);
	setTetrahedronVertices(m_tet[3], m_d, m_red[1], m_blue[3], m_blue[1]);
	
	setTetrahedronVertices(m_tet[4], m_red[0], m_blue[2], m_blue[0], m_blue[1]);
	setTetrahedronVertices(m_tet[5], m_red[0], m_blue[2], m_blue[1], m_blue[3]);
	setTetrahedronVertices(m_tet[6], m_blue[2], m_red[1], m_blue[0], m_blue[1]);
	setTetrahedronVertices(m_tet[7], m_blue[2], m_red[1], m_blue[1], m_blue[3]);
	
	m_N = 8;
}

void RedBlueRefine::twoBlueUpRefine()
{
	bool lhs = m_blue[0] > -1;
	if(lhs) {
		setTetrahedronVertices(m_tet[0], m_a, m_b, m_blue[0], m_blue[1]);
		setTetrahedronVertices(m_tet[1], m_c, m_red[1], m_blue[0], m_b);
		setTetrahedronVertices(m_tet[2], m_blue[0], m_blue[1], m_b, m_red[1]);
		setTetrahedronVertices(m_tet[3], m_d, m_red[1], m_b, m_blue[1]);
	
	}
	else {
		setTetrahedronVertices(m_tet[0], m_a, m_b, m_blue[2], m_blue[3]);
		setTetrahedronVertices(m_tet[1], m_c, m_red[1], m_a, m_blue[2]);
		setTetrahedronVertices(m_tet[2], m_blue[2], m_blue[3], m_red[1], m_a);
		setTetrahedronVertices(m_tet[3], m_d, m_red[1], m_blue[3], m_a);
	
	}
	m_N = 4;
}
	
void RedBlueRefine::twoBlueDownRefine()
{
	bool lhs = m_blue[0] > -1;
	if(lhs) {
		setTetrahedronVertices(m_tet[0], m_a, m_red[0], m_blue[0], m_d);
		setTetrahedronVertices(m_tet[1], m_blue[0], m_blue[2], m_c, m_d);
		setTetrahedronVertices(m_tet[2], m_blue[0], m_blue[2], m_d, m_red[0]);
		setTetrahedronVertices(m_tet[3], m_red[0], m_b, m_blue[2], m_d);
	
	}
	else {
		setTetrahedronVertices(m_tet[0], m_a, m_red[0], m_c, m_blue[1]);
		setTetrahedronVertices(m_tet[1], m_blue[1], m_blue[3], m_c, m_d);
		setTetrahedronVertices(m_tet[2], m_blue[1], m_blue[3], m_red[0], m_c);
		setTetrahedronVertices(m_tet[3], m_red[0], m_b, m_c, m_blue[3]);
	
	}
	m_N = 4;
}

}
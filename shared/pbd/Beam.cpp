/*
 *  Beam.cpp
 *  
 *
 *  Created by jian zhang on 7/29/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "Beam.h"
#include <geom/SegmentNormals.h>
#include <math/MatrixC33F.h>
#include <math/linspace.h>

namespace aphid {

namespace pbd {

Beam::Beam() : m_numSegs(0)
{ m_ref.set(-1.f, 0.f, 0.f); }

Beam::~Beam()
{}

void Beam::setGhostRef(const Vector3F& ref)
{ m_ref = ref; }

void Beam::createNumSegments(int spp)
{
	const int ns = spp * 3;
	m_p.reset(new Vector3F[ns + 1]);
	m_gp.reset(new Vector3F[ns]);
	m_invMass.reset(new float[ns + 1]);
	m_stiffness.reset(new float[ns + 1]);
	m_constraintSegInd.reset(new int[ns]);
	m_numSegs = ns;
	
	permutateConstraintInd();
	calculatePnts();
/// default inv_mass curve
	HermiteInterpolatePiecewise<float, Vector2F > invMassCurve;
	static const float pnt[4][2] = {
	{0.f, .5f},
	{4.f, 1.f},
	{8.f, 3.f},
	{12.f, 5.f},
	};
	static const float tng[4][2] = {
	{1.f, 0.f},
	{1.f, 0.f},
	{1.f, 0.1f},
	{1.f, 0.1f},
	};
	for(int i=0;i<3;++i) {
		invMassCurve.setPieceBegin(i, pnt[i], tng[i]);
		invMassCurve.setPieceEnd(i, pnt[i+1], tng[i+1]);
	}
	calculateInvMass(&invMassCurve);
	
	static const float pntStiff[4][2] = {
	{0.f, 1.f},
	{4.f, .8f},
	{8.f, .4f},
	{12.f, .1f},
	};
	static const float tngStiff[4][2] = {
	{1.f, -.1f},
	{1.f, -.1f},
	{1.f, 0.f},
	{1.f, 0.f},
	};
	for(int i=0;i<3;++i) {
		invMassCurve.setPieceBegin(i, pntStiff[i], tngStiff[i]);
		invMassCurve.setPieceEnd(i, pntStiff[i+1], tngStiff[i+1]);
	}
	calculateStiffness(&invMassCurve);
	
}

void Beam::calculateInvMass(HermiteInterpolatePiecewise<float, Vector2F > * crv)
{
	interpolateValue(m_invMass.get(), crv);
}

void Beam::calculateStiffness(HermiteInterpolatePiecewise<float, Vector2F > * crv)
{
	interpolateValue(m_stiffness.get(), crv);
}

void Beam::interpolateValue(float* vals,
		HermiteInterpolatePiecewise<float, Vector2F > * crv)
{
	const int& ns = m_numSegs;
	const int spp = ns / 3;
	const int ppp = spp + 1;
	float* param = new float[ppp];
	linspace<float>(param, 0.f, 1.f, ppp);
	
	int acc = 0;
	for(int j=0;j<3;++j) {
		for(int i=0;i<spp;++i) {
			vals[acc++] = crv->interpolate(j, param[i]).y;
		}
	}
	
	delete[] param;
	
/// last particle
	vals[acc] = crv->Pnt(5).y;
}

void Beam::calculatePnts()
{
	const int& ns = m_numSegs;
	const int spp = ns / 3;
	const int ppp = spp + 1;
	float* param = new float[ppp];
	linspace<float>(param, 0.f, 1.f, ppp);
	
	int acc = 0;
	for(int j=0;j<3;++j) {
		for(int i=0;i<spp;++i) {
			m_p[acc++] = interpolate(j, param[i]);
		}
	}
	
	delete[] param;
	
/// last particle
	m_p[acc] = Pnt(5);
	
	SegmentNormals segnml(ns);
	Vector3F p0p1 = m_p[1] - m_p[0];
	segnml.calculateFirstNormal(p0p1, m_ref);
	m_gp[0] = (m_p[0] + m_p[1]) * .5f + segnml.getNormal(0);
	
	for(int i=1;i<ns;++i) {
		Vector3F p0 = m_p[i - 1];
		Vector3F p1 = m_p[i];
		Vector3F p2 = m_p[i+1];
		p0p1 = p1 - p0;
		Vector3F p1p2 = p2 - p1;
		Vector3F p1pm02 = (p0 + p2) * 0.5f - p1;
		segnml.calculateNormal(i, p0p1, p1p2, p1pm02 );
/// mid edge rest length?
		m_gp[i] = (p1 + p2) * .5f + segnml.getNormal(i) * p1.distanceTo(p2) * .5f;
	
	}
	
}

const int& Beam::numSegments() const
{ return m_numSegs; }

const int Beam::numParticles() const
{ return m_numSegs + 1; }

const int& Beam::numGhostParticles() const
{ return m_numSegs; }

const Vector3F& Beam::getParticlePnt(int i) const
{ return m_p[i]; }

const Vector3F& Beam::getGhostParticlePnt(int i) const
{ return m_gp[i]; }

const float& Beam::getStiffness(int i) const
{ return m_stiffness[i]; }

const float& Beam::getInvMass(int i) const
{ return m_invMass[i]; }	

const int& Beam::getConstraintSegInd(int i) const
{ return m_constraintSegInd[i]; }

Vector3F Beam::getSegmentMidPnt(int i) const
{ return (m_p[i] + m_p[i+1]) * .5f; }

void Beam::permutateConstraintInd()
{
	for(int i=0;i<m_numSegs;i++) {
		m_constraintSegInd[i] = i;
	}
	
	int j = m_numSegs - 1;
	for(int i=1;i<m_numSegs;i+=2) {
		m_constraintSegInd[j] = i;
		m_constraintSegInd[i] = j;
		j-=2;
		if(i>=j) break;
	}
	
}

MatrixC33F Beam::getMaterialFrame(int i) const
{
	Vector3F vz = m_p[i+1] - m_p[i];
	vz.normalize();
	
	Vector3F vx = m_gp[i] - getSegmentMidPnt(i);
	
	Vector3F vy = vz.cross(vx);
	MatrixC33F frm;
	frm.setCol(0, vx);
	frm.setCol(1, vy);
	frm.setCol(2, vz);
	return frm;
}

}

}
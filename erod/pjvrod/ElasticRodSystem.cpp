/*
 *  ElasticRodSystem.cpp
 *  
 *  https://github.com/millag/DiscreteElasticRods/blob/master/src/ElasticRod.cpp
 *
 *  Created by jian zhang on 1/6/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "ElasticRodSystem.h"
#include <geom/ParallelTransport.h>
#include <math/Matrix33F.h>
#include <math/miscfuncs.h>

namespace aphid {
namespace pbd {

ElasticRodSystem::ElasticRodSystem() : m_numSegments(0) {}

ElasticRodSystem::~ElasticRodSystem() {}

int ElasticRodSystem::numNodes() const
{ return m_numSegments + 1; }

const int& ElasticRodSystem::numSegments() const
{ return m_numSegments; }

const Vector3F* ElasticRodSystem::positions() const
{ return m_pos.get(); }

void ElasticRodSystem::getBishopFrame(Vector3F& c,
				Vector3F& t, Vector3F& u, Vector3F& v,
				int i) const
{
	c = (m_pos[i] + m_pos[i+1]) * .5f;
	t = m_kb[i];
	u = m_m1[i];
	v = m_m2[i];
}

void ElasticRodSystem::create()
{	
	m_pos.reset(new Vector3F[32]);
	for(int i=0;i<32;++i) {
		m_pos[i].set(10.f + 1.5f * i, 10.f - .5f * i + 2.f * sin(0.75f * i), 2.3f * cos(.85f * i) );
	}
	
	m_numSegments = 31;
	
	m_edges.reset(new Vector3F[m_numSegments]);
	m_kb.reset(new Vector3F[m_numSegments]);
	m_m1.reset(new Vector3F[m_numSegments]);
	m_m2.reset(new Vector3F[m_numSegments]);
	m_restEdgeL.reset(new float[m_numSegments]);
	m_restRegionL.reset(new float[m_numSegments]);
	m_restWprev.reset(new Vector2F[m_numSegments]);
	m_restWnext.reset(new Vector2F[m_numSegments]);
	computeEdges();
	computeLengths();
/// up of first rod
	const Vector3F u0(0.f, 1.f, 0.f);
	computeCurvatureBinormals();
	m_u0 = u0;
	computeBishopFrame();
	computeMaterialCurvature();
	updateCurrentState();
	
}

void ElasticRodSystem::computeEdges()
{
	for(int i=0;i<m_numSegments;++i) {
		m_edges[i] = m_pos[i + 1] - m_pos[i];
	}
}

void ElasticRodSystem::computeLengths()
{
	for(int i=0;i<m_numSegments;++i) {
		m_restEdgeL[i] = m_edges[i].length();
	}
	
	m_restRegionL[0] = 0.f;
	for(int i=1;i<m_numSegments;++i) {
		m_restRegionL[i] = m_restEdgeL[i - 1] + m_restEdgeL[i];
	}
}

void ElasticRodSystem::computeCurvatureBinormals()
{
	m_kb[0].setZero();
	for(int i=1;i<m_numSegments;++i) {
		m_kb[i] = m_edges[i - 1].cross(m_edges[i]) * (2.f / (m_restEdgeL[i - 1] * m_restEdgeL[i] + m_edges[i - 1].dot(m_edges[i])) );
	}
}

void ElasticRodSystem::extractSinAndCos(float& sinPhi, float& cosPhi,
			const float& kdk) const
{
	cosPhi = sqrt(4.f / (4.f + kdk));
	sinPhi = sqrt(kdk / (4.f + kdk));	
}

void ElasticRodSystem::computeBishopFrame()
{
	m_m1[0] = m_u0;
	m_m2[0] = m_edges[0].cross(m_u0);
	m_m2[0].normalize();
	
	m_m1[0] = m_m2[0].cross(m_edges[0]);
	m_m1[0].normalize();
	
	Matrix33F frm;
	frm.fill(m_m2[0], m_m1[0], m_edges[0].normal() );
	
	float kdk, sinPhi, cosPhi;
	for(int i=1;i<m_numSegments;++i) {
		kdk = m_kb[i].dot(m_kb[i]);
		extractSinAndCos(sinPhi, cosPhi, kdk);
		
		if(1.f - cosPhi < 1e-5f) {
/// straight
			m_m1[i] = m_m1[i - 1];
			m_m2[i] = m_m2[i - 1];
			continue;
		}
		
		//Quaternion q(cosPhi, m_kb[i].normal() * sinPhi);
		float ang = acos(m_edges[i-1].normal().dot(m_edges[i].normal() ) );
		Quaternion q(ang, m_kb[i].normal());
		Matrix33F rot(q);
		frm *= rot;

		m_m1[i].set(frm.M(1,0), frm.M(1,1), frm.M(1,2) );
		//Quaternion p(.1f, m_m1[i - 1]);
		//p = q * p;
		//m_m1[i].set(p.x, p.y, p.z);
		m_m1[i].normalize();
		
		m_m2[i] = m_edges[i].cross(m_m1[i]);
		m_m2[i].normalize();
		
	}
}

void ElasticRodSystem::computeMaterialCurvature()
{
	m_restWprev[0].setZero();
	m_restWnext[0].setZero();
	for (unsigned i = 1; i <m_numSegments; ++i)
	{
		computeW(m_restWprev[i], m_kb[i], m_m1[i - 1], m_m2[i - 1]);
		computeW(m_restWnext[i], m_kb[i], m_m1[i], m_m2[i]);
	}
}

void ElasticRodSystem::computeW(Vector2F& dst, const Vector3F& kb, 
						const Vector3F& m1, const Vector3F& m2) const
{
	dst.set(kb.dot(m2), -kb.dot(m1) );
}

void ElasticRodSystem::updateCurrentState()
{
	Vector3F e0 = m_edges[0];
	computeEdges();
	Vector3F e1 = m_edges[0];
	Vector3F u0 = m_u0;
	ParallelTransport::Rotate(u0, e0, e1);
	m_u0 = u0;
	computeBishopFrame();
	
}

}
}
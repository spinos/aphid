#include "ParticleData.h"

namespace aphid {
namespace pbd {
    
ParticleData::ParticleData() :
m_posLast(0),
m_pos(0),
m_projectedPos(0),
m_force(0),
m_velocity(0),
m_Ri(0),
m_geomNml(0),
m_invMass(0),
m_localGeomNml(0),
m_numParticles(0)
{}

ParticleData::~ParticleData()
{
    if(m_posLast) delete[] m_posLast;
    if(m_pos) delete[] m_pos;
    if(m_projectedPos) delete[] m_projectedPos;
    if(m_force) delete[] m_force;
    if(m_velocity) delete[] m_velocity;
    if(m_Ri) delete[] m_Ri;
	if(m_geomNml) delete[] m_geomNml;
    if(m_invMass) delete[] m_invMass;
	if(m_localGeomNml) delete[] m_localGeomNml;
}

void ParticleData::createNParticles(int x)
{
    m_posLast = new Vector3F[x];
	m_pos = new Vector3F[x];
	m_projectedPos = new Vector3F[x];
	m_force = new Vector3F[x];
	m_velocity = new Vector3F[x];
	m_Ri = new Vector3F[x];
	m_geomNml = new Vector3F[x];
	m_invMass = new float[x];
	m_localGeomNml = new char[x];
	m_numParticles = x;
}

const int& ParticleData::numParticles() const
{ return m_numParticles; }

const Vector3F* ParticleData::pos() const
{ return m_pos; }

Vector3F* ParticleData::pos()
{ return m_pos; }

Vector3F* ParticleData::projectedPos()
{ return m_projectedPos; }

Vector3F* ParticleData::posLast()
{ return m_posLast; }

Vector3F * ParticleData::force()
{ return m_force; }

Vector3F * ParticleData::velocity()
{ return m_velocity; }

Vector3F * ParticleData::Ri()
{ return m_Ri; }

Vector3F* ParticleData::geomNml()
{ return m_geomNml; }

float * ParticleData::invMass()
{ return m_invMass; }

char* ParticleData::localGeomNml()
{ return m_localGeomNml; }

void ParticleData::setParticle(const Vector3F& pv, int i)
{
    m_posLast[i] = pv;
    m_pos[i] = pv;
    m_projectedPos[i] = pv;
    m_force[i].setZero();
    m_velocity[i].setZero();
    m_Ri[i].setZero();
    m_invMass[i] = 1.f;
    m_localGeomNml[i] = 0;
	m_geomNml[i].set(0,1,0);
}

void ParticleData::cachePositions()
{
	for(int i=0;i<m_numParticles;++i) {
		m_posLast[i] = m_pos[i];
		m_pos[i] = m_projectedPos[i];
	}
}

void ParticleData::dampVelocity(float damping)
{
	if(damping < 1e-3f) return;
	
	float sz = 1.f - damping;
	for(int i=0;i<m_numParticles;++i) {
		m_velocity[i] *= sz;
	}
}

void ParticleData::projectPosition(float dt)
{
	for(int i=0;i< m_numParticles;i++) {
	    if(m_invMass[i] > 0.f) m_projectedPos[i] = m_pos[i] + m_velocity[i] * dt;
	}
}

void ParticleData::updateVelocityAndPosition(float dt)
{
	for(int i=0;i< m_numParticles;i++) {
	    m_velocity[i] = (m_projectedPos[i] - m_pos[i]) / dt;
	    m_pos[i] = m_projectedPos[i];
	}
}

void ParticleData::zeroVelocity()
{
	memset(m_velocity, 0, m_numParticles*12);
}

}
}

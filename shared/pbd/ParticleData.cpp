#include "ParticleData.h"

namespace aphid {
namespace pbd {
    
ParticleData::ParticleData() :
m_pos(0),
m_projectedPos(0),
m_posLast(0),
m_force(0),
m_velocity(0),
m_Ri(0),
m_invMass(0),
m_numParticles(0)
{}

ParticleData::~ParticleData()
{
    if(m_pos) delete[] m_pos;
    if(m_projectedPos) delete[] m_projectedPos;
    if(m_posLast) delete[] m_posLast;
    if(m_force) delete[] m_force;
    if(m_velocity) delete[] m_velocity;
    if(m_Ri) delete[] m_Ri;
    if(m_invMass) delete[] m_invMass;
}

void ParticleData::createNParticles(int x)
{
    m_pos = new Vector3F[x];
	m_posLast = new Vector3F[x];
	m_force = new Vector3F[x];
	m_invMass = new float[x];
	m_velocity = new Vector3F[x];
	m_Ri = new Vector3F[x];
	m_projectedPos = new Vector3F[x];
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

float * ParticleData::invMass()
{ return m_invMass; }

void ParticleData::setParticle(const Vector3F& pv, int i)
{
    m_pos[i] = pv;
    m_projectedPos[i] = pv;
    m_posLast[i] = pv;
    m_force[i].setZero();
    m_velocity[i].setZero();
    m_Ri[i].setZero();
    m_invMass[i] = 1.f;
    
}

}
}


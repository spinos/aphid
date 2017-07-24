#include "SimulationContext.h"

namespace aphid {
namespace pbd {
    
SimulationContext::SimulationContext() :
m_pos(0),
m_projectedPos(0),
m_posLast(0),
m_force(0),
m_velocity(0),
m_Ri(0),
m_invMass(0),
m_numPoints(0)
{}

SimulationContext::~SimulationContext()
{
    if(m_pos) delete[] m_pos;
    if(m_projectedPos) delete[] m_projectedPos;
    if(m_posLast) delete[] m_posLast;
    if(m_force) delete[] m_force;
    if(m_velocity) delete[] m_velocity;
    if(m_Ri) delete[] m_Ri;
    if(m_invMass) delete[] m_invMass;
}

void SimulationContext::createNPoints(int x)
{
    m_pos = new Vector3F[x];
	m_posLast = new Vector3F[x];
	m_force = new Vector3F[x];
	m_invMass = new float[x];
	m_velocity = new Vector3F[x];
	m_Ri = new Vector3F[x];
	m_projectedPos = new Vector3F[x];
	m_numPoints = x;
}

const int& SimulationContext::numPoints() const
{ return m_numPoints; }

Vector3F* SimulationContext::pos()
{ return m_pos; }

Vector3F* SimulationContext::projectedPos()
{ return m_projectedPos; }

Vector3F* SimulationContext::posLast()
{ return m_posLast; }

Vector3F * SimulationContext::force()
{ return m_force; }

Vector3F * SimulationContext::velocity()
{ return m_velocity; }

Vector3F * SimulationContext::Ri()
{ return m_Ri; }

float * SimulationContext::invMass()
{ return m_invMass; }

/// http://www.physics.udel.edu/~bnikolic/teaching/phys660/numerical_ode/node5.html
/// http://codeflow.org/entries/2010/aug/28/integration-by-example-euler-vs-verlet-vs-runge-kutta/
/// https://gamedevelopment.tutsplus.com/tutorials/simulate-tearable-cloth-and-ragdolls-with-simple-verlet-integration--gamedev-519
/// http://wiki.roblox.com/index.php?title=Verlet_integration
void SimulationContext::integrateVerlet(float deltaTime) 
{
	float deltaTime2 = deltaTime*deltaTime;

	for(int i=0;i< m_numPoints;i++) {
		Vector3F buffer = m_pos[i];
		m_pos[i] = m_pos[i] + (m_pos[i] - m_posLast[i]) + m_force[i] * deltaTime2 / m_invMass[i];
		m_posLast[i] = buffer;
	}
}

void SimulationContext::integrate(float deltaTime) 
{	
	float inv_dt = 1.f/deltaTime;

	for(int i=0;i< m_numPoints;i++) {	
		m_velocity[i] = (m_projectedPos[i] - m_pos[i])*inv_dt;		
		m_pos[i] = m_projectedPos[i];		 
	}
}
    
}
}


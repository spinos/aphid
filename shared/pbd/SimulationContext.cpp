#include "SimulationContext.h"

namespace aphid {
namespace pbd {
    
SimulationContext::SimulationContext()
{}

SimulationContext::~SimulationContext()
{}

const ParticleData* SimulationContext::c_particles() const
{ return &m_part; }

ParticleData* SimulationContext::particles()
{ return &m_part; }

/// http://www.physics.udel.edu/~bnikolic/teaching/phys660/numerical_ode/node5.html
/// http://codeflow.org/entries/2010/aug/28/integration-by-example-euler-vs-verlet-vs-runge-kutta/
/// https://gamedevelopment.tutsplus.com/tutorials/simulate-tearable-cloth-and-ragdolls-with-simple-verlet-integration--gamedev-519
/// http://wiki.roblox.com/index.php?title=Verlet_integration
void SimulationContext::integrateVerlet(float deltaTime) 
{
    const int& np = m_part.numParticles();
    Vector3F* x = m_part.pos();
    Vector3F* xLast = m_part.posLast();
    Vector3F* f = m_part.force();
    float* im = m_part.invMass();
	float deltaTime2 = deltaTime*deltaTime;

	for(int i=0;i< np;i++) {
		Vector3F buffer = x[i];
		x[i] = x[i] + (x[i] - xLast[i]) + f[i] * deltaTime2 / im[i];
		xLast[i] = buffer;
	}
}

void SimulationContext::integrate(float deltaTime) 
{	
	float inv_dt = 1.f/deltaTime;
	const int& np = m_part.numParticles();
    Vector3F* x = m_part.pos();
    Vector3F* xProj = m_part.projectedPos();
    Vector3F* v = m_part.velocity();
	for(int i=0;i< np;i++) {	
		v[i] = (xProj[i] - x[i])*inv_dt;		
		x[i] = xProj[i];		 
	}
}

void SimulationContext::clearGravitiyForce()
{
    static const Vector3F gravity(0.f, -980.f, 0.f);
    const int& np = m_part.numParticles();
    Vector3F* f = m_part.force();
    float* im = m_part.invMass();
    for(int i=0;i< np;i++) {
		f[i].setZero();
		if(im[i] > 0.f) {
		    f[i] += gravity / im[i];
		}
	}
}

void SimulationContext::semiImplicitEulerIntegrate(ParticleData* part, float dt)
{
	const int& np = part->numParticles();
	Vector3F* vel = part->velocity();
	Vector3F* pos = part->projectedPos();
	Vector3F* f = part->force();
    float* im = part->invMass();
	
	for(int i=0;i< np;i++) {
		vel[i] += f[i] * (im[i] * dt);
		pos[i] += vel[i] * dt;
	}
}

void SimulationContext::addExternalForce()
{}

void SimulationContext::positionConstraintProjection()
{}

}
}


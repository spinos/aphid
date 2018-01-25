#include "SimulationContext.h"
#include "WindForce.h"
#include <math/miscfuncs.h>
#include <lbm/VolumeResponse.h>
#include <lbm/LatticeBlock.h>

namespace aphid {
namespace pbd {
    
SimulationContext::SimulationContext()
{
	m_latman = new lbm::VolumeResponse;	
	m_meanWindVel[0] = -1.f;
	m_meanWindVel[1] = 0.f;
	m_meanWindVel[2] = 0.f;
	m_gravityY = -98.f;
	m_isCollisionOn = true;
}

SimulationContext::~SimulationContext()
{}

const bool& SimulationContext::isCollisionEnabled() const
{ return m_isCollisionOn; }

void SimulationContext::enableCollision()
{ m_isCollisionOn = true; }

void SimulationContext::disableCollision()
{ m_isCollisionOn = false; }

const ParticleData* SimulationContext::c_particles() const
{ return &m_part; }

ParticleData* SimulationContext::particles()
{ return &m_part; }

const ParticleData* SimulationContext::c_ghostParticles() const
{ return 0; }

ParticleData* SimulationContext::ghostParticles()
{ return 0; }

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

void SimulationContext::applyGravity(float dt)
{ applyGravityTo(&m_part, dt); }

void SimulationContext::applyGravityTo(ParticleData* part, float dt)
{
    const int& np = part->numParticles();
	Vector3F* vel = part->velocity();
	const float* im = part->invMass();
	for(int i=0;i< np;i++) {
	    if(im[i] > 0.f) vel[i].y += m_gravityY * dt;
	}
}

void SimulationContext::applyWind(float dt)
{
	applyWindTo(&m_part, dt);
}

void SimulationContext::applyWindTo(ParticleData* part, float dt)
{
	Vector3F vair(m_meanWindVel);
	if(vair.length2() < 1e-3f) return;
	
	const int& np = part->numParticles();
	Vector3F* vel = part->velocity();
	Vector3F* nml = part->geomNml();
	const float* im = part->invMass();
	for(int i=0;i< np;i++) {
	    if(im[i] > 0.f) {
			Vector3F relair = vair - vel[i];
			Vector3F fdl = WindForce::ComputeDragAndLift(relair, nml[i]);
			vel[i] += fdl * (im[i] * dt);
		}
	}
}

void SimulationContext::projectPosition(float dt)
{
    m_part.projectPosition(dt);
}

void SimulationContext::updateVelocityAndPosition(float dt)
{
    m_part.updateVelocityAndPosition(dt);
}

void SimulationContext::dampVelocity(float damping)
{ m_part.dampVelocity(damping); } 

Vector3F SimulationContext::getGravityVec() const
{ return Vector3F(0.f, m_gravityY, 0.f); }

const float& SimulationContext::grivityY() const
{ return m_gravityY; }

void SimulationContext::setMeanWindVelocity(const Vector3F& vwind)
{ 
	m_meanWindVel[0] = vwind.x;
	m_meanWindVel[1] = vwind.y;
	m_meanWindVel[2] = vwind.z;
}

void SimulationContext::applyCollisionConstraint()
{
	if(!m_isCollisionOn)
		return;
		
	const int& np = m_part.numParticles();
	Vector3F* vel = m_part.velocity();
	Vector3F* x = m_part.pos();
	m_latman->solveParticles((float*)vel, (const float*)x, np);
}

void SimulationContext::resetCollisionGrid(const float& cellSize)
{
	std::cout<<"\n SimulationContext::resetCollisionGrid cell size "<<cellSize;
	lbm::LatticeParam param;
	param._blockSize = cellSize * 16.f;
/// 30 fps 4 substeps 
/// dx/dt <- velocity / 120
	param._inScale = .0083f;
	param._outScale = 120.f;
	m_latman->setParam(param);
}

}
}


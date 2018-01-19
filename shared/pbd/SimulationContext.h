/*
 *  Position-Based-Dynamics
 *  holding all point properties
 */
#ifndef APH_PBD_SIMULATIONCONTEXT_H
#define APH_PBD_SIMULATIONCONTEXT_H

#include "ParticleData.h"
#include <vector>

namespace aphid {

namespace lbm {

class VolumeResponse;

}

namespace pbd {

class SimulationContext {

    ParticleData m_part;
    float m_meanWindVel[3];
	float m_gravityY;
	lbm::VolumeResponse* m_latman;
	
public:
    SimulationContext();
    virtual ~SimulationContext();
    
    ParticleData* particles();
    const ParticleData* c_particles() const;
	virtual ParticleData* ghostParticles();
	virtual const ParticleData* c_ghostParticles() const;
	
protected:
    void integrateVerlet(float dt);
    void integrate(float dt);
/// clear force add gravity
    void clearGravitiyForce();
/// v <- v + g dt
    virtual void applyGravity(float dt);
	virtual void applyWind(float dt);
/// v <- v + a dt
/// x <- x + v dt
	void semiImplicitEulerIntegrate(ParticleData* part, float dt);
	virtual void addExternalForce();
/// modify v at x     
	virtual void applyCollisionConstraint();
	virtual void positionConstraintProjection();
/// x* <- x + v dt
    virtual void projectPosition(float dt);
/// v <- (x* - x) / dt
/// x <- x*
    virtual void updateVelocityAndPosition(float dt);
/// v <- v damping
    virtual void dampVelocity(float damping);
	
	void applyGravityTo(ParticleData* part, float dt);
	void applyWindTo(ParticleData* part, float dt);
	
	Vector3F getGravityVec() const;
	const float& grivityY() const;
	
	void setMeanWindVelocity(const Vector3F& vwind);
/// cell size	
	void resetCollisionGrid(const float& cellSize);
	
private:
   
};

}
}

#endif        //  #ifndef APH_PBD_SIMULATIONCONTEXT_H


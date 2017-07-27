/*
 *  Position-Based-Dynamics
 *  holding all point properties
 */
#ifndef APH_PBD_SIMULATIONCONTEXT_H
#define APH_PBD_SIMULATIONCONTEXT_H

#include "ParticleData.h"
#include <vector>

namespace aphid {
namespace pbd {

class SimulationContext {

    ParticleData m_part;
    
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
/// v <- v + a dt
/// x <- x + v dt
	void semiImplicitEulerIntegrate(ParticleData* part, float dt);
	virtual void addExternalForce();
	virtual void positionConstraintProjection();
/// x* <- x + v dt
    virtual void projectPosition(float dt);
/// v <- (x* - x) / dt
/// x <- x*
    virtual void updateVelocityAndPosition(float dt);
/// v <- v damping
    virtual void dampVelocity(float damping);
    
private:
   
};

}
}

#endif        //  #ifndef APH_PBD_SIMULATIONCONTEXT_H


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
    
    const ParticleData* c_particles() const;
    
protected:
    ParticleData* particles();
    void integrateVerlet(float dt);
    void integrate(float dt);
/// clear force add gravity
    void clearGravitiyForce();
	
private:
   
};

}
}

#endif        //  #ifndef APH_PBD_SIMULATIONCONTEXT_H


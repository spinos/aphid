/*
 *  Position-Based-Dynamics
 *  holding all point properties
 */
#ifndef APH_PBD_SIMULATIONCONTEXT_H
#define APH_PBD_SIMULATIONCONTEXT_H

#include <math/Vector3F.h>

namespace aphid {
namespace pbd {
class SimulationContext {

    Vector3F * m_pos;
	Vector3F * m_projectedPos;
	Vector3F * m_posLast;
	Vector3F * m_force;
	Vector3F * m_velocity;
	Vector3F * m_Ri;
	float * m_invMass;
	int m_numPoints;
	
public:
    SimulationContext();
    virtual ~SimulationContext();
    
    const int& numPoints() const;
    
    Vector3F* pos();
    Vector3F* projectedPos();
    Vector3F* posLast();
    Vector3F * force();
	Vector3F * velocity();
	Vector3F * Ri();
	float * invMass();
	
protected:
    void createNPoints(int x);
    void integrateVerlet(float dt);
    void integrate(float dt);
	
private:
};
}
}

#endif        //  #ifndef APH_PBD_SIMULATIONCONTEXT_H


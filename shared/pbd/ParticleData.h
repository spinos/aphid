#ifndef APH_PBD_PARTICLE_DATA_H
#define APH_PBD_PARTICLE_DATA_H

#include <math/Vector3F.h>

namespace aphid {
namespace pbd {
 
class ParticleData {

    Vector3F * m_posLast;
	Vector3F * m_pos;
	Vector3F * m_projectedPos;
	Vector3F * m_force;
	Vector3F * m_velocity;
	Vector3F * m_Ri;
	float * m_invMass;
	int m_numParticles;
	
public:
    ParticleData();
    virtual ~ParticleData();
    
    void createNParticles(int x);
    const int& numParticles() const;
    
    void setParticle(const Vector3F& pv, int i);
    
    const Vector3F* pos() const;
    Vector3F* pos();
    Vector3F* projectedPos();
    Vector3F* posLast();
    Vector3F* force();
	Vector3F* velocity();
	Vector3F* Ri();
	float * invMass();
	
/// posLast <- pos
/// pos<- posProjected
	void cachePositions();
/// v <- v(1 - d)
	void dampVelocity(float damping);
/// x* <- x + v dt
    void projectPosition(float dt);
/// v <- (x* - x) / dt
/// x <- x*
    void updateVelocityAndPosition(float dt);
    
private:
};

}
}

#endif        //  #ifndef PARTICLEDATA_H


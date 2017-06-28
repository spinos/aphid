#ifndef SOLVERTHREAD_H
#define SOLVERTHREAD_H

#include <qt/BaseSolverThread.h>
#include <pbd_common.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE
//class BoxProgram;
class SolverThread : public aphid::BaseSolverThread
{
    //BoxProgram * m_program;
    aphid::Vector3F * m_pos;
	aphid::Vector3F * m_projectedPos;
	aphid::Vector3F * m_posLast;
	aphid::Vector3F * m_force;
	aphid::Vector3F * m_velocity;
	unsigned * m_indices;
	float * m_invMass;
	aphid::Vector3F * m_Ri;
	pbd::Spring * m_spring;
	pbd::DistanceConstraint * m_distanceConstraint;
	unsigned m_numBendingConstraint, m_numDistanceConstraint;
	
public:
    SolverThread(QObject *parent = 0);
    ~SolverThread();
    
    void initProgram();
	
    unsigned numIndices() const;
    aphid::Vector3F * pos();
    unsigned * indices();

protected:
    virtual void stepPhysics(float dt);
	
private:
    void setSpring(pbd::Spring * dest, unsigned a, unsigned b, float ks, float kd, int type);
	void setDistanceConstraint(pbd::DistanceConstraint * dest, unsigned a, unsigned b, float k);
	void computeForces();
	void integrateExplicitWithDamping(float dt);
	void integrateVerlet(float dt);
	void updateConstraints(float dt);
	void updateDistanceConstraint(unsigned i);
	void groundCollision();
	void integrate(float deltaTime);
	
	static aphid::Vector3F getVerletVelocity(aphid::Vector3F x_i, aphid::Vector3F xi_last, float dt );

public slots:
   
};

#endif        //  #ifndef SOLVERTHREAD_H


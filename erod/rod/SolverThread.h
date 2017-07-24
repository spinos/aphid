/*
 *  rodt
 */
#ifndef SOLVERTHREAD_H
#define SOLVERTHREAD_H

#include <qt/BaseSolverThread.h>
#include <pbd/pbd_common.h>
#include <pbd/SimulationContext.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class SolverThread : public aphid::BaseSolverThread, public aphid::pbd::SimulationContext
{
	unsigned * m_indices;
	aphid::pbd::Spring * m_spring;
	aphid::pbd::DistanceConstraint * m_distanceConstraint;
	unsigned m_numBendingConstraint, m_numDistanceConstraint;
	
public:
    SolverThread(QObject *parent = 0);
    ~SolverThread();
    
    void initProgram();
	
    unsigned numIndices() const;
    unsigned * indices();

protected:
    virtual void stepPhysics(float dt);
	
private:
    void setSpring(aphid::pbd::Spring * dest, unsigned a, unsigned b, float ks, float kd, int type);
	void setDistanceConstraint(aphid::pbd::DistanceConstraint * dest, unsigned a, unsigned b, float k);
	void computeForces();
	void integrateExplicitWithDamping(float dt);
	void updateConstraints(float dt);
	void updateDistanceConstraint(unsigned i);
	void groundCollision();
	
	static aphid::Vector3F getVerletVelocity(aphid::Vector3F x_i, aphid::Vector3F xi_last, float dt );

public slots:
   
};

#endif        //  #ifndef SOLVERTHREAD_H


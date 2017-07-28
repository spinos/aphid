/*
 *  rodt
 */
#ifndef SOLVERTHREAD_H
#define SOLVERTHREAD_H

#include <qt/BaseSolverThread.h>
#include <pbd/pbd_common.h>
#include <pbd/ElasticRodContext.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class SolverThread : public aphid::BaseSolverThread, public aphid::pbd::ElasticRodContext
{
	unsigned * m_indices;
	aphid::pbd::Spring * m_spring;
	aphid::pbd::DistanceConstraint * m_distanceConstraint;
	unsigned m_numBendingConstraint, m_numDistanceConstraint;
	
public:
    SolverThread(QObject *parent = 0);
    ~SolverThread();

protected:
    virtual void stepPhysics(float dt);
	
private:
    void createBeam(const aphid::Matrix44F& tm);
    void createBones();
    
public slots:
   
};

#endif        //  #ifndef SOLVERTHREAD_H


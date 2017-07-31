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

namespace aphid {

namespace pbd {
class WindTurbine;
}
}

class SolverThread : public aphid::BaseSolverThread, public aphid::pbd::ElasticRodContext
{
	unsigned * m_indices;
	aphid::pbd::Spring * m_spring;
	aphid::pbd::DistanceConstraint * m_distanceConstraint;
	unsigned m_numBendingConstraint, m_numDistanceConstraint;
	
public:
    SolverThread(QObject *parent = 0);
    ~SolverThread();
	
	aphid::pbd::WindTurbine* windTurbine();
	const aphid::pbd::WindTurbine* windTurbine() const;

protected:
    virtual void stepPhysics(float dt);
	
private:
	aphid::pbd::WindTurbine* m_windicator;
	
private:
    void createBeam(const aphid::Matrix44F& tm);
    void createBones();
    
public slots:
   
};

#endif        //  #ifndef SOLVERTHREAD_H


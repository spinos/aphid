/*
 *  projective rod
 *
 */
#ifndef TEST_SOLVER_H
#define TEST_SOLVER_H

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

class TestSolver : public aphid::pbd::ElasticRodContext, public aphid::BaseSolverThread
{
	unsigned * m_indices;
	aphid::pbd::Spring * m_spring;
	aphid::pbd::DistanceConstraint * m_distanceConstraint;
	unsigned m_numBendingConstraint, m_numDistanceConstraint;
	
public:
    TestSolver(QObject *parent = 0);
    ~TestSolver();
	
	aphid::pbd::WindTurbine* windTurbine();
	const aphid::pbd::WindTurbine* windTurbine() const;
	
protected:
    virtual void stepPhysics(float dt);
    
private:
	aphid::pbd::WindTurbine* m_windicator;
	
private:
    void createBeam(const aphid::Matrix44F& tm);
    
};

#endif        //  #ifndef TestSolver_H


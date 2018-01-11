/*
 *  projective rod
 *
 */
#ifndef TEST_SOLVER_H
#define TEST_SOLVER_H

#include <qt/BaseSolverThread.h>
#include <pbd/pbd_common.h>
#include <pbd/ShapeMatchingContext.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

namespace aphid {

namespace pbd {
class WindTurbine;
}
}

class TestSolver : public aphid::pbd::ShapeMatchingContext, public aphid::BaseSolverThread
{
	
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
    
};

#endif        //  #ifndef TestSolver_H


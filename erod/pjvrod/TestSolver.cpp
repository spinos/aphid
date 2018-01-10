/*
 *  projective rod
 *  projective implicit euler solver
 *  reference http://www.projectivedynamics.org/projectivedynamics.pdf
 *
 */
#include <QtCore>
#include "TestSolver.h"
#include <pbd/Beam.h>
#include <pbd/WindTurbine.h>
#include <smp/UniformGrid8Sphere.h>

using namespace aphid;

TestSolver::TestSolver(QObject *parent)
    : BaseSolverThread(parent)
{
	m_windicator = new pbd::WindTurbine;

    Matrix33F rtm;
	rtm = rtm * .43f;
    Matrix44F spc;
    spc.setRotation(rtm);
    spc.setTranslation(8,4,2);

	create();
}

TestSolver::~TestSolver()
{
}

void TestSolver::stepPhysics(float dt)
{
#if 1
    applyGravity(dt);
	setMeanWindVelocity(m_windicator->getMeanWindVec() );
	applyWind(dt);
	projectPosition(dt);	
	positionConstraintProjection();
	dampVelocity(0.02f);
	updateVelocityAndPosition(dt);
#endif	
	m_windicator->progress(dt);
	BaseSolverThread::stepPhysics(dt);
}

pbd::WindTurbine* TestSolver::windTurbine()
{ return m_windicator; }

const pbd::WindTurbine* TestSolver::windTurbine() const
{ return m_windicator; }

void TestSolver::restartCurrentState()
{ 
	std::cout<<" TestSolver::restartCurrentState"<<std::endl; 
	ghostParticles()->zeroVelocity();
	particles()->zeroVelocity();
	
}

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
#include <pbd/ShapeMatchingProfile.h>

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

	pbd::ShapeMatchingProfile prof;
	prof.createTestStrand();
	create(prof);
	
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
	applyCollisionConstraint();
	projectPosition(dt);
	updateShapeMatchingRegions();	
	positionConstraintProjection();
	updateVelocityAndPosition(dt);
	//dampVelocity(0.001f);
#endif	
	m_windicator->progress(dt);
	BaseSolverThread::stepPhysics(dt);
}

pbd::WindTurbine* TestSolver::windTurbine()
{ return m_windicator; }

const pbd::WindTurbine* TestSolver::windTurbine() const
{ return m_windicator; }

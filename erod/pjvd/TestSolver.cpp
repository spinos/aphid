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

	Vector3F g[33];
	for(int i=0;i<33;++i) {
		g[i].set(10.f + (1.199f - .00863 * i) * i, 
				21.f + 2.98f * (.5f - .005f * i) * sin(0.75f * i), 
				0.f + 2.99f * (.5f - .005f * i) * cos(.85f * i) );
	}
	
	pbd::ShapeMatchingProfile prof;
	prof.clearStrands();
	for(int i=0;i<33;++i) {
		prof.addStrandPoint(g[i]);
	}
	pbd::StrandParam sparam;
	sparam._binormal.set(0.f, 1.f, 0.f);
	sparam._mass0 = .5f;
	sparam._mass1 = .4f;
	sparam._stiffness0 = 1.f;
	sparam._stiffness1 = 1.f;
	prof.finishStrand(sparam);
	
	for(int i=0;i<33;++i) {
		g[i].set(10.f + (1.197f - .0093 * i) * i,
				20.5f + 2.98f * (.5f - .0027f * i) * sin(0.474f * i - .53f), 
				-.5f + 2.98f * (.51f - .0026f * i) * cos(.55f + .49f * i) );
	}
	
	for(int i=0;i<33;++i) {
		prof.addStrandPoint(g[i]);
	}
	
	prof.finishStrand(sparam);
	
	for(int i=0;i<32;++i) {
		g[i].set(10.f + (1.193f - .00933 * i) * i, 
				20.5f + 2.98f * (.5f - .0025f * i) * sin(0.64f * i + .532f), 
				.5f + 2.98f * (.5f - .0025f * i) * cos(.43f * i - .351f) );
	}
	
	for(int i=0;i<32;++i) {
		prof.addStrandPoint(g[i]);
	}
	
	prof.finishStrand(sparam);
	
	for(int i=0;i<33;++i) {
		g[i].set(10.f + (1.297f - .0103 * i) * i, 
				19.5f + 2.98f * (.45f - .0013f * i) * sin(0.6f * i - .45f), 
				-.5f + 2.98f * (.46f - .0013f * i) * cos(.53f * i - .35f) );
	}
	
	for(int i=0;i<33;++i) {
		prof.addStrandPoint(g[i]);
	}
	
	prof.finishStrand(sparam);
	
	for(int i=0;i<33;++i) {
		g[i].set(10.f + (1.197f - .00993 * i) * i,
				19.5f + 2.98f * (.45f - .0013f * i) * sin(.54f * i + .74f),
				.5f + 2.98f * (.35f - .0025f * i) * cos(0.544f * i + .65f) );
	}
	
	for(int i=0;i<33;++i) {
		prof.addStrandPoint(g[i]);
	}
	
	prof.finishStrand(sparam);

	for(int i=0;i<32;++i) {
		g[i].set(10.f + (1.297f - .0113 * i) * i,
				18.5f + 2.98f * (.45f - .0015f * i) * sin(0.734f * i + .587f), 
				0.f + 2.98f * (.45f - .0015f * i) * cos(.5413f * i + .45f) );
	}
	
	for(int i=0;i<32;++i) {
		prof.addStrandPoint(g[i]);
	}
	
	prof.finishStrand(sparam);
		
	prof.buildProfile();
	prof.setLod(1.2f);
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
	if(isCollisionEnabled() )
		applyCollisionConstraint();
	projectPosition(dt);
	updateShapeMatchingRegions();	
	positionConstraintProjection();
	updateVelocityAndPosition(dt);
#endif	
	m_windicator->progress(dt);
	BaseSolverThread::stepPhysics(dt);
}

pbd::WindTurbine* TestSolver::windTurbine()
{ return m_windicator; }

const pbd::WindTurbine* TestSolver::windTurbine() const
{ return m_windicator; }

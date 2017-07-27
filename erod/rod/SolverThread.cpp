/*
 *  rodt
 */
#include <QtCore>
#include "SolverThread.h"

using namespace aphid;

#define NU 24
#define NP (NU + 1)

SolverThread::SolverThread(QObject *parent)
    : BaseSolverThread(parent)
{
    pbd::ParticleData* part = particles();
	part->createNParticles(NP);
	for(int i=0;i<NP;++i) {
        part->setParticle(Vector3F(1.f * i, 2.f, 0.f), i);
    }
    
	pbd::ParticleData* ghost = ghostParticles();
	ghost->createNParticles(NP-1);
    for(int i=0;i<NP-1;++i) {
        ghost->setParticle(Vector3F(1.f * i + .5f, 3.f, 0.f), i);
    }
    
///lock two first particles and first ghost point
    part->invMass()[0] = 0.f;
    part->invMass()[1] = 0.f;
    ghost->invMass()[0] = 0.f;
    
	for(int i=0;i<NP-1;++i) {
	    addElasticRodEdgeConstraint(i, i+1, i);
	}
	
	for(int i=0;i<NP-2;++i) {
	    addElasticRodBendAndTwistConstraint(i, i+1, i+2, i, i+1);
	}
	
	createEdges();
	
}

SolverThread::~SolverThread()
{
}

void SolverThread::stepPhysics(float dt)
{
    applyGravity(dt);
    projectPosition(dt);	
	positionConstraintProjection();
	dampVelocity(0.01f);
	updateVelocityAndPosition(dt);
	
	BaseSolverThread::stepPhysics(dt);
}

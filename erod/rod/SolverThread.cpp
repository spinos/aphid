/*
 *  rodt
 */
#include <QtCore>
#include "SolverThread.h"
#include "bones.h"

using namespace aphid;

SolverThread::SolverThread(QObject *parent)
    : BaseSolverThread(parent)
{
#if 0
    Matrix33F rtm;
    rtm.rotateZ(1.5f);
    rtm.rotateX(.55f);
    Matrix44F spc;
    spc.setRotation(rtm);
    spc.setTranslation(1,2,3);
    createBeam(1.f, 0.034f, spc);
#else 
    createBones();
#endif
	createEdges();
}

SolverThread::~SolverThread()
{
}

void SolverThread::stepPhysics(float dt)
{
#if 1
    applyGravity(dt);
    projectPosition(dt);	
	positionConstraintProjection();
	dampVelocity(0.01f);
	updateVelocityAndPosition(dt);
#endif	
	BaseSolverThread::stepPhysics(dt);
}

void SolverThread::createBeam(float deltaL, float deltaAngle, const Matrix44F& tm)
{
#define NU 18
#define NP (NU + 1)

    Vector3F dP(deltaL,0,0), curP(0,0,0);
    Matrix33F rotm;
    pbd::ParticleData* part = particles();
	part->createNParticles(NP);
	for(int i=0;i<NP;++i) {
        part->setParticle(tm.transform(curP), i);
        rotm.rotateZ(deltaAngle);
        curP += rotm.transform(dP);
    }
    
    rotm.setIdentity();
    rotm.rotateZ(deltaAngle * 0.5f);
	curP.set(deltaL * .5f, 1.f,0.f);
    pbd::ParticleData* ghost = ghostParticles();
	ghost->createNParticles(NP-1);
    for(int i=0;i<NP-1;++i) {
        ghost->setParticle(tm.transform(curP), i);
        rotm.rotateZ(deltaAngle);
        curP += rotm.transform(dP);
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
}

void SolverThread::createBones()
{
    pbd::ParticleData* part = particles();
	part->createNParticles(sFooNumParticles);
	for(int i=0;i<sFooNumParticles;++i) {
        part->setParticle(Vector3F(sFooParticlePoints[i]), i);
    }
    
    pbd::ParticleData* ghost = ghostParticles();
	ghost->createNParticles(sFooNumGhostParticles);
	for(int i=0;i<sFooNumGhostParticles;++i) {
        ghost->setParticle(Vector3F(sFooGhostParticlePoints[i]), i);
    }
    
    part->invMass()[0] = 0.f;
    part->invMass()[1] = 0.f;
    ghost->invMass()[0] = 0.f;
    
    for(int i=0;i<sFooNumEdgeConstraints;++i) {
        const int* ind = sFooEdgeConstraints[i];
	    addElasticRodEdgeConstraint(ind[0], ind[1], ind[2]);
	}
	
	for(int i=0;i<sFooNumBendAndTwistConstraints;++i) {
	    const int* ind = sFooBendAndTwistConstraints[i];
	    addElasticRodBendAndTwistConstraint(ind[0], ind[1], ind[2], ind[3], ind[4]);
	}
}


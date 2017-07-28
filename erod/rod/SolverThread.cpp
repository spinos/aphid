/*
 *  rodt
 */
#include <QtCore>
#include "SolverThread.h"
#include "bones.h"
#include <pbd/Beam.h>

using namespace aphid;

SolverThread::SolverThread(QObject *parent)
    : BaseSolverThread(parent)
{
#if 1
    Matrix33F rtm;
    rtm.rotateZ(-2.5f);
	//rtm.rotateY(-1.5f);
    //rtm.rotateX(1.55f);
    Matrix44F spc;
    //spc.setRotation(rtm);
    spc.setTranslation(4,5,6);
    createBeam(spc);
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

void SolverThread::createBeam(const Matrix44F& tm)
{
static const float P0[4][3] = {
 {1.4973,0.0135717,0},
 {3.61696,27.5391,-0.852699},
 {4.40409,48.8,-2.99834},
 {-8.09004,60.4578,-8.07755},
};

static const float T0[4][3] = {
 {-0.176762,16.6883,-0.949179},
 {6.95258,24.9088,-0.236556},
 {-5.12741,20.8168,-3.41737},
 {-7.79407,4.84709,-8.56933},
};
	pbd::Beam bem;
	for(int i=0;i<3;++i) {
		bem.setPieceBegin(i, P0[i], T0[i]);
		bem.setPieceEnd(i, P0[i+1], T0[i+1]);
	}
	bem.createNumSegments(4);
	
	const int np = bem.numParticles();
	
    pbd::ParticleData* part = particles();
	part->createNParticles(np);
	for(int i=0;i<np;++i) {
        part->setParticle(tm.transform(bem.getParticlePnt(i) ), i);
    }
    
	const int& ngp = bem.numGhostParticles();
	
	pbd::ParticleData* ghost = ghostParticles();
	ghost->createNParticles(ngp);
    for(int i=0;i<ngp;++i) {
        ghost->setParticle(tm.transform(bem.getGhostParticlePnt(i) ), i);
    }
    
///lock two first particles and first ghost point
    part->invMass()[0] = 0.f;
    part->invMass()[1] = 0.f;
    ghost->invMass()[0] = 0.f;
    
	const int& ns = bem.numSegments();
	std::cout<<"\n nseg "<<ns;
	for(int i=0;i<ns;++i) {
		const int& ci = bem.getConstraintSegInd(i);
	    addElasticRodEdgeConstraint(ci, ci+1, ci);
		std::cout<<"\n eg "<<ci<<" "<<(ci+1)<<" "<<ci;
	}
	
	for(int i=0;i<ns;++i) {
		const int& ci = bem.getConstraintSegInd(i);
		if(ci < ns - 1) {
			addElasticRodBendAndTwistConstraint(ci, ci+1, ci+2, ci, ci+1);
			std::cout<<"\n bnt "<<ci<<" "<<(ci+1)<<" "<<(ci+2)<<" "<<(ci)<<" "<<(ci+1);
		}
	}
	std::cout.flush();
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


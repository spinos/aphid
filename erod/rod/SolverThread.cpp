/*
 *  rodt
 */
#include <QtCore>
#include "SolverThread.h"

using namespace aphid;

#define NU 12
#define NV 32
#define NF (NU * NV)
#define NTri (NF * 2)
#define NI (NTri * 3)
#define NP (NU + 1)

#define STRUCTURAL_SPRING 0
#define SHEAR_SPRING 1
#define BEND_SPRING 2

const float DEFAULT_DAMPING =  -0.0825f;
float	KsStruct = 180.5f,KdStruct = -.25f;
float	KsShear = 180.5f,KdShear = -.25f;
float	KsBend = 80.5f,KdBend = -.25f;

Vector3F gravity(0.0f,-981.f,0.0f);
float mass = 1.f;
float iniHeight = 99.f;
float gridSize = iniHeight / (float)NU;
const unsigned solver_iterations = 7;
float kBend = 0.5f;
float kStretch = 1.f; 
float kDamp = 0.025f;
float global_dampening = 0.99f;

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
 /*   
    clearGravitiyForce();
	addExternalForce();
	pbd::ParticleData* part = particles();
	part->cachePositions();
	part->dampVelocity(0.0f);
	
	semiImplicitEulerIntegrate(part, dt);
	
	pbd::ParticleData* ghost = ghostParticles();
	ghost->cachePositions();
	ghost->dampVelocity(0.0f);
	
	semiImplicitEulerIntegrate(ghost, dt);
*/	
	positionConstraintProjection();
	dampVelocity(0.01f);
	updateVelocityAndPosition(dt);
	//integrateExplicitWithDamping(dt);
	//updateConstraints(dt);
	//integrate(dt);
	
	BaseSolverThread::stepPhysics(dt);
}

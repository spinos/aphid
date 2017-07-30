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
    rtm.rotateZ(-1.5f);
	//rtm.rotateY(-1.5f);
    //rtm.rotateX(1.55f);
    Matrix44F spc;
    //spc.setRotation(rtm);
    spc.setTranslation(6,5,4);
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
	applyWind(dt);
    projectPosition(dt);	
	positionConstraintProjection();
	dampVelocity(0.01f);
	updateVelocityAndPosition(dt);
#endif	
	BaseSolverThread::stepPhysics(dt);
}

void SolverThread::createBeam(const Matrix44F& tm)
{
static const float P0[16][3] = {
 {1.4973,0.0135717,0},
 {3.61696,27.5391,-0.852699},
 {4.40409,48.8,-2.99834},
 {-8.09004,60.4578,-8.07755},
 {0.239419,0,0},
 {2.26399,11.6731,-0.852699},
 {5.80086,25.8274,-1.50135},
 {14.6939,31.0103,-4.02684},
 {-0.883503,0.0963957,0},
 {-1.36914,12.6204,-0.759043},
 {-3.4397,27.3021,0.0450491},
 {-7.02952,39.8786,-2.8443},
  {-0.91866,-2.87294e-05,0},
 {-3.39053,11.7276,0.151154},
 {-9.40395,20.0547,0.945781},
 {-16.8805,20.8402,2.60076},
 
};

static const float T0[16][3] = {
 {-0.176762,16.6883,-0.949179},
 {6.95258,24.9088,-0.236556},
 {-5.12741,20.8168,-3.41737},
 {-7.79407,4.84709,-8.56933},
{1.16552,8.85645,0.118592},
 {0.213102,14.175,-1.12544},
 {6.97878,8.44556,-1.91793},
 {8.12793,2.29141,-1.25547},
 {0.147823,11.6085,0.118592},
 {-2.4618,14.1684,-1.12544},
 {-0.721688,12.5583,-1.91793},
 {-7.35406,8.41031,-1.25547},
 {0.147823,9.43875,-0.103063},
 {-2.4618,10.4636,0.567969},
 {-6.41507,2.97874,1.18756},
 {-5.13812,-1.64422,4.80791},
 
};

#define BmNumMdl 3
#define BmMdlOffset 4

	pbd::Beam bem[BmNumMdl];
	for(int j=0;j<BmNumMdl;++j) {
		const int ll = BmMdlOffset + j * 4;
		for(int i=0;i<3;++i) {
			Vector3F pnt = tm.transform(Vector3F(P0[ll + i]) );
			Vector3F tng = tm.transformAsNormal(Vector3F(T0[ll + i]) );
			bem[j].setPieceBegin(i, pnt, tng);
			
			pnt = tm.transform(Vector3F(P0[ll + i + 1]) );
			tng = tm.transformAsNormal(Vector3F(T0[ll + i + 1]) );
			bem[j].setPieceEnd(i, pnt, tng);
		}
		
		if(j==2) {
			bem[j].setGhostRef(Vector3F(1,0,0));
	
		}
		bem[j].createNumSegments(3);
		
	}
	
	
	createBeams(bem, BmNumMdl);
	
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
	    addElasticRodBendAndTwistConstraint(ind[0], ind[1], ind[2], ind[3], ind[4], 1.f);
	}
}


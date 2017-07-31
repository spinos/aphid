/*
 *  darboux
 */
#include "TestContext.h"
#include "TestConstraint.h"
#include <pbd/ParticleData.h>
#include <math/Matrix44F.h>
#include <math/MatrixC33F.h>

using namespace aphid;

#define NP 3

TestContext::TestContext()
{
    pbd::ParticleData* part = particles();
	part->createNParticles(NP);
	for(int i=0;i<NP;++i) {
        part->setParticle(Vector3F(2.f * i, 2.f, 0.f), i);
    }
    
	pbd::ParticleData* ghost = ghostParticles();
	ghost->createNParticles(NP-1);
    for(int i=0;i<NP-1;++i) {
        ghost->setParticle(Vector3F(2.f * i + 1.f, 3.f, 0.f), i);
    }
    
///lock two first particles and first ghost point
    //part->invMass()[0] = 0.f;
    //part->invMass()[1] = 0.f;
    //ghost->invMass()[0] = 0.f;
    
	for(int i=0;i<NP-1;++i) {
	    addElasticRodEdgeConstraint(i, i+1, i);
	}
	
	for(int i=0;i<NP-2;++i) {
	    //addElasticRodBendAndTwistConstraint(i, i+1, i+2, i, i+1);
		pbd::TestConstraint *cn = new pbd::TestConstraint();
		const bool res = cn->initConstraint(this, i, i+1, i+2, i, i+1);
		addElasticRodBendAndTwistConstraint(cn);
	}
	
	createEdges();
	
	const Vector3F a(0,1,0);
	Matrix33F ma;
	ma.asCrossProductMatrix(a);
	std::cout<<"\n test cross product [a]"<<ma;
	
	const Vector3F b(0,0,1);
	Matrix33F mbt;
	mbt.asCrossProductMatrix(b);
	mbt.transpose();

	Vector3F maxb = ma * b;
	Vector3F mbtxa = mbt * a;
	
	std::cout<<"\n answer a x b = "<<a.cross(b)
		<<"\n [a]xb"<<maxb
		<<"\n [b]^txa"<<mbtxa;
		
	std::cout<<"\n test MatrixC33F ";
	MatrixC33F mca;
	mca.asCrossProductMatrix(a);
	std::cout<<"\n cross product [a]"<<mca;
	
	MatrixC33F mcbt;
	mcbt.asCrossProductMatrix(b);
	mcbt.transpose();
	
	Vector3F mcaxb = mca * b;
	Vector3F mcbtxa = mcbt * a;
	
	std::cout<<"\n answer a x b = "<<a.cross(b)
		<<"\n [a]xb"<<mcaxb
		<<"\n [b]^txa"<<mcbtxa;
		
	MatrixC33F tim;
	tim.setZero();
	tim.addDiagonal(4.f);
	
	std::cout<<"\n test inverse a"<<tim
		<<"\n a^-1"<<tim.inversed();
}

TestContext::~TestContext()
{
}

void TestContext::stepPhysics(float dt)
{
}

void TestContext::getMaterialFrames(Matrix44F& frmA, Matrix44F& frmB,
							Vector3F& darboux,
							Vector3F* correctVs)
{
	pbd::ElasticRodBendAndTwistConstraint* c = bendAndTwistConstraint(0);
	pbd::TestConstraint* tc = static_cast<pbd::TestConstraint* > (c);
	tc->getMaterialFrames(frmA, frmB, darboux, correctVs, this);
	
}

void TestContext::rotateFrame(const Quaternion& rot)
{
	Matrix33F mrot(rot);
	pbd::ParticleData* part = particles();
	Vector3F dc = part->pos()[2] - part->pos()[1];
	dc = mrot * dc;
	part->pos()[2] = part->pos()[1] + dc;
	
	pbd::ParticleData* ghost = ghostParticles();
	Vector3F de = ghost->pos()[1] - part->pos()[1];
	de = mrot * de;
	ghost->pos()[1] = part->pos()[1] + de;
	
}

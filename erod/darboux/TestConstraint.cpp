/*
 *  TestConstraint.cpp
 *  
 *
 *  Created by jian zhang on 7/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "TestConstraint.h"
#include <pbd/ParticleData.h>
#include <pbd/SimulationContext.h>
#include <math/Matrix44F.h>
#include <math/MatrixC33F.h>

namespace aphid  {

namespace pbd {

TestConstraint::TestConstraint()
{}

void TestConstraint::getMaterialFrames(Matrix44F& frmA, Matrix44F& frmB, 
					Vector3F& darboux, Vector3F* correctVs,
					SimulationContext * model)
{
	ParticleData* part = model->particles();
	ParticleData* ghost = model->ghostParticles();
	Vector3F* ps = part->pos();
    Vector3F* gs = ghost->pos();
	const float* pm = part->invMass();
    const float* gm = ghost->invMass();
    const int* inds = c_bodyInds();
    Vector3F& xA = ps[inds[0]];
    Vector3F& xB = ps[inds[1]];
	Vector3F& xC = ps[inds[2]];
	Vector3F& xD = gs[inds[3]];
	Vector3F& xE = gs[inds[4]];
	
	const float wA = pm[inds[0]];
	const float wB = pm[inds[1]];
	const float wC = pm[inds[2]];
	const float wD = gm[inds[3]];
	const float wE = gm[inds[4]];
	
	MatrixC33F dA, dB;

	computeMaterialFrame(dA, xA, xB, xD);
	computeMaterialFrame(dB, xB, xC, xE);
	computeDarbouxVector(darboux, dA, dB, 1.f);
	
	Matrix33F mA;
	mA.fill(dA.colV(0), dA.colV(1), dA.colV(2));
	Matrix33F mB;
	mB.fill(dB.colV(0), dB.colV(1), dB.colV(2));
	
	frmA.setRotation(mA);
	frmA.setTranslation((xA + xB) * .5f);
	frmB.setRotation(mB);
	frmB.setTranslation((xB + xC) * .5f);
	
	MatrixC33F dajpi[3][3];
	computeMaterialFrameDerivative(xA, xB, xD, dA,
		dajpi[0][0], dajpi[0][1], dajpi[0][2],
		dajpi[1][0], dajpi[1][1], dajpi[1][2],
		dajpi[2][0], dajpi[2][1], dajpi[2][2]);
#if 0		
	std::cout<<"\n dA"<<dA
	<<"\n d1dp0"<<dajpi[0][0]
	<<"\n d1dp1"<<dajpi[0][1]
	<<"\n d1dp2"<<dajpi[0][2]
	<<"\n d2dp0"<<dajpi[1][0]
	<<"\n d2dp1"<<dajpi[1][1]
	<<"\n d2dp2"<<dajpi[1][2]
	<<"\n d3dp0"<<dajpi[2][0]
	<<"\n d3dp1"<<dajpi[2][1]
	<<"\n d3dp2"<<dajpi[2][2]
	<<"\n";
#endif
	MatrixC33F  dbjpi[3][3];
	computeMaterialFrameDerivative(xB, xC, xE, dB,
		dbjpi[0][0], dbjpi[0][1], dbjpi[0][2],
		dbjpi[1][0], dbjpi[1][1], dbjpi[1][2],
		dbjpi[2][0], dbjpi[2][1], dbjpi[2][2]);
#if 0	
	std::cout<<"\n dB"
	<<"\n d1dp0"<<dbjpi[0][0]
	<<"\n d1dp1"<<dbjpi[0][1]
	<<"\n d1dp2"<<dbjpi[0][2]
	<<"\n d2dp0"<<dbjpi[1][0]
	<<"\n d2dp1"<<dbjpi[1][1]
	<<"\n d2dp2"<<dbjpi[1][2]
	<<"\n d3dp0"<<dbjpi[2][0]
	<<"\n d3dp1"<<dbjpi[2][1]
	<<"\n d3dp2"<<dbjpi[2][2]
	<<"\n";
#endif	

	MatrixC33F constraint_jacobian[5];
	computeDarbouxGradient(darboux, 1.f, dA, dB, 
		dajpi, dbjpi, 
		Vector3F(1,1,1),
		constraint_jacobian[0],
		constraint_jacobian[1],
		constraint_jacobian[2],
		constraint_jacobian[3],
		constraint_jacobian[4]);
#if 0
	for(int i=0;i<5;++i) {
		std::cout<<"\n constraint_jacobian"<<i
			<<" "<<constraint_jacobian[i];
	}
#endif
	
	Vector3F constraint_value = darboux - restDarbouxVector();
	
	MatrixC33F factor_matrix;
	factor_matrix.setZero();
	
	MatrixC33F tmp_mat;
	float invMasses[] = { wA, wB, wC, wD, wE };
	for (int i = 0; i < 5; ++i) {
		tmp_mat = constraint_jacobian[i].transposed() * constraint_jacobian[i];
		tmp_mat *= invMasses[i];

		factor_matrix += tmp_mat;
	}
	
	tmp_mat = factor_matrix.inversed();

#if 0	
	std::cout<<"\n factor mat"<<factor_matrix
		<<"\n inv "<<tmp_mat;
#endif

	for (int i = 0; i < 5; ++i) {
        constraint_jacobian[i] *= invMasses[i];
        correctVs[i] = constraint_jacobian[i] * (tmp_mat * constraint_value);
        correctVs[i] *= -1.f;
		
		std::cout<<"\n dp"<<i<<" "<<correctVs[i];
    }
		
	std::cout.flush();

}

}

}
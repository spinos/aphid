/*
 *  ElasticRodAttachmentConstraint.cpp
 *  
 *
 *  Created by jian zhang on 1/7/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "ElasticRodAttachmentConstraint.h"
#include "SimulationContext.h"
#include "ParticleData.h"
#include <math/Quaternion.h>
#include <math/miscfuncs.h>

namespace aphid {
namespace pbd {

ElasticRodAttachmentConstraint::ElasticRodAttachmentConstraint()
{}

ElasticRodAttachmentConstraint::ConstraintType ElasticRodAttachmentConstraint::getConstraintType() const
{ return ctElasticRodAttachment; }

bool ElasticRodAttachmentConstraint::initConstraint(SimulationContext * model, const int pA, const int pB, const int pC,
                                    const int pD, const int pE)
{
	bodyInds()[0] = pA;
    bodyInds()[1] = pB;
    bodyInds()[2] = pC;
    bodyInds()[3] = pD;
    bodyInds()[4] = pE;
	
	const Vector3F* ps = model->c_particles()->pos();
    const Vector3F* gs = model->c_ghostParticles()->pos();
    const Vector3F& xA = ps[pA];
	const Vector3F& xB = ps[pB];
	const Vector3F& xC = ps[pC];
	const Vector3F& xD = gs[pD];
	const Vector3F& xE = gs[pE];
	
/// material frame ABD
	MatrixC33F dA; 
/// material frame BCE
	MatrixC33F dB;
	computeMaterialFrame(dA, xA, xB, xD);
	computeMaterialFrame(dB, xB, xC, xE);
	
	float l = (xA - xB).length();
	setEdgeRestLength(l );
	computeDarbouxVector(restDarbouxVector(), dA, dB, l);
	setBendAndTwistKs(1.f, 1.f, 1.f);
	
	m_p0 = xA;
	m_p1 = xB;
	m_g0 = xD;
	
	return true;
}

bool ElasticRodAttachmentConstraint::solvePositionConstraint(ParticleData* part, ParticleData* ghost)
{
	Vector3F* ps = part->projectedPos();
    Vector3F* gs = ghost->projectedPos();
    const float* pm = part->invMass();
    const float* gm = ghost->invMass();
    const int* inds = c_bodyInds();
    Vector3F& xA = ps[inds[0]];
    Vector3F& xB = ps[inds[1]];
    Vector3F& xC = ps[inds[2]];
	Vector3F& xD = ps[inds[3]];
	Vector3F& xE = gs[inds[4]];
	
	const float wA = pm[inds[0]];
	const float wB = pm[inds[1]];
	const float wC = pm[inds[2]];
	const float wD = gm[inds[3]];
	const float wE = gm[inds[4]];
	
	Vector3F corr[5];
	bool res = projectBendingAndTwistingConstraint(
		m_p0, wA, m_p1, wB, xC, wC, m_g0, wD, xE, wE, 
		bendAndTwistKs(), midEdgeRestLength(), restDarbouxVector(), 
		corr[0], corr[1], corr[2], corr[3], corr[4]);
	//return true;
	if (res) {
			xC += corr[2] * stiffness();

	}
	return res;
}

void ElasticRodAttachmentConstraint::updateConstraint(ParticleData* part, ParticleData* ghost)
{
	Vector3F* ps = part->projectedPos();
    Vector3F* gs = ghost->projectedPos();
	const int* inds = c_bodyInds();
    Vector3F& xA = ps[inds[0]];
    Vector3F& xB = ps[inds[1]];
    Vector3F& xD = ps[inds[3]];
	
	m_p0 = xA;
	m_p1 = xB;
	m_g0 = xD;
}

}
}


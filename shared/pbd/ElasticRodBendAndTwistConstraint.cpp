#include "ElasticRodBendAndTwistConstraint.h"
#include "SimulationContext.h"
#include "ParticleData.h"
#include <math/miscfuncs.h>

namespace aphid {
namespace pbd {
    
ElasticRodBendAndTwistConstraint::ElasticRodBendAndTwistConstraint()
{}

ElasticRodBendAndTwistConstraint::ConstraintType ElasticRodBendAndTwistConstraint::getConstraintType() const
{ return ctElasticRodBendAndTwist; }

bool ElasticRodBendAndTwistConstraint::initConstraint(SimulationContext * model, const int pA, const int pB, const int pC,
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
	
	computeMaterialFrame(m_dA, xA, xB, xD);
	computeMaterialFrame(m_dB, xB, xC, xE);
	computeDarbouxVector(m_restDarbouxVector, m_dA, m_dB, 1.0f);
	m_bendAndTwistKs.set(1.f,1.f,1.f);
    return true;
}

bool ElasticRodBendAndTwistConstraint::solvePositionConstraint(ParticleData* part, ParticleData* ghost)
{
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
	
	Vector3F corr[5];
	bool res = projectBendingAndTwistingConstraint(
		xA, wA, xB, wB, xC, wC, xD, wD, xE, wE, 
		m_bendAndTwistKs, 1.0f, m_restDarbouxVector, 
		corr[0], corr[1], corr[2], corr[3], corr[4]);
	if (res) {
		const float stiffness = 1.f;//model.getElasticRodBendAndTwistStiffness();
		if (wA != 0.0f)
			xA += corr[0] * stiffness;
		
		if (wB != 0.0f)
			xB += corr[1] * stiffness;
		
		if (wC != 0.0f)
			xC += corr[2] * stiffness;
		
		if (wD != 0.0f)
			xD += corr[3] * stiffness;

		if (wE != 0.0f)
			xE += corr[4] * stiffness;
	}
	return res;
}

bool ElasticRodBendAndTwistConstraint::projectBendingAndTwistingConstraint(const Vector3F& pA, const float wA, 
		const Vector3F& pB, const float wB, 
		const Vector3F& pC, const float wC,
		const Vector3F& pD, const float wD,
		const Vector3F& pE, const float wE,
		const Vector3F& bendingAndTwistingKs,
		const float midEdgeLength,
		const Vector3F& restDarbouxVector,
		Vector3F& corrA, Vector3F& corrB, Vector3F& corrC, Vector3F& corrD, Vector3F& corrE)
{
    return true;
}

///     G
///   /
///  A ---- B
void ElasticRodBendAndTwistConstraint::computeMaterialFrame(Matrix33F& frame,
	        const Vector3F& vA, const Vector3F& vB, const Vector3F& vG)
{
    Vector3F vz = vB - vA;
    vz.normalize();
    
    Vector3F vy = vz.cross(vG - vA);
    vy.normalize();
    
    Vector3F vx = vy.cross(vz);
    frame.fill(vx, vy, vz);
}

static const int permutation[3][3] = {
	0, 2, 1,
	1, 0, 2,
	2, 1, 0
};

void ElasticRodBendAndTwistConstraint::computeDarbouxVector(Vector3F& darboux,
	        const Matrix33F& frameA, const Matrix33F& frameB,
	        float midEdgeLength)
{
    float factor = 1.0f + frameA.row(0).dot(frameB.row(0))
                        + frameA.row(1).dot(frameB.row(1)) 
                        + frameA.row(2).dot(frameB.row(2));
	factor = 2.0f / (midEdgeLength * factor);
	for (int c = 0; c < 3; ++c) {
		const int i = permutation[c][0];
		const int j = permutation[c][1];
		const int k = permutation[c][2];

		darboux.setComp(frameA.row(j).dot(frameB.row(k) ) - frameA.row(k).dot(frameB.row(j) ), 
		                i);
	}

	darboux *= factor;
}
    
}
}

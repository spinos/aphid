#include "ElasticRodEdgeConstraint.h"
#include "SimulationContext.h"
#include "ParticleData.h"
#include <math/miscfuncs.h>

namespace aphid {
namespace pbd {
    
ElasticRodEdgeConstraint::ElasticRodEdgeConstraint()
{}

ElasticRodEdgeConstraint::ConstraintType ElasticRodEdgeConstraint::getConstraintType() const
{ return ctElasticRodEdge; }

bool ElasticRodEdgeConstraint::initConstraint(SimulationContext * model, const int pA, const int pB, const int pG)
{
    bodyInds()[0] = pA;
    bodyInds()[1] = pB;
    bodyInds()[2] = pG;
    
    const Vector3F* ps = model->c_particles()->pos();
    const Vector3F* gs = model->c_ghostParticles()->pos();
    
    const Vector3F& xA = ps[pA];
    const Vector3F& xB = ps[pB];
    const Vector3F& xG = gs[pG];
    
    m_restLength = xA.distanceTo(xB);
    m_ghostRestLength = xG.distanceTo((xA + xB) * .5f );
    m_edgeKs = 1.f;
    return true;
}

bool ElasticRodEdgeConstraint::solvePositionConstraint(ParticleData* part, ParticleData* ghost)
{ 
	const int iA = c_bodyInds()[0];
	const int iB = c_bodyInds()[1];
	const int iG = c_bodyInds()[2];
	
	Vector3F& xA = part->projectedPos()[iA];
	Vector3F& xB = part->projectedPos()[iB];
	Vector3F& xG = ghost->projectedPos()[iG];
	
	float wA = part->invMass()[iA];
	float wB = part->invMass()[iB];
	float wG = ghost->invMass()[iG];
	
	Vector3F corr[3];
	const bool res = projectEdgeConstraints(xA, wA, xB, wB, xG, wG, 
				m_edgeKs, m_restLength, m_ghostRestLength, 
				corr[0], corr[1], corr[2]);
	if (res) {
		if (wA != 0.0f)
			xA += corr[0];
		if (wB != 0.0f)
			xB += corr[1];
		if (wG != 0.0f)
			xG += corr[2];
	}
	
	return res;
}

bool ElasticRodEdgeConstraint::projectEdgeConstraints(
		const Vector3F& pA, const float wA, 
		const Vector3F& pB, const float wB, 
		const Vector3F& pG, const float wG,
		const float edgeKs, const float edgeRestLength, const float ghostEdgeRestLength,
		Vector3F& corrA, Vector3F& corrB, Vector3F& corrC)
{
	corrA.setZero(); corrB.setZero(); corrC.setZero();
	Vector3F dir = pA - pB;
	float len = dir.length();
	float wSum = wA + wB;
	if (len > EPSILON && wSum > EPSILON) {
		Vector3F dP = dir * ( (1.0f / wSum) * (len - edgeRestLength) * edgeKs / len );
		corrA -= dP * wA;
		corrB += dP * wB;
		corrC = Vector3F(0, 0, 0);
	}
	
	Vector3F pm = (pA + pB) * .5f;
	Vector3F p0p2 = pA - pG;
	Vector3F p2p1 = pG - pB;
	Vector3F p1p0 = pB - pA;
	Vector3F p2pm = pG - pm;
	float lambda;
	
	wSum = wA * p0p2.length2() + wB * p2p1.length2() + wG * p1p0.length2();
	if (wSum > EPSILON) {
		lambda = p2pm.dot(p1p0) / wSum * edgeKs;
		corrA -= p0p2 * lambda * wA;
		corrB -= p2p1 * lambda * wB;
		corrC -= p1p0 * lambda * wG;
	}
	
	wSum = 0.25f * wA + 0.25f * wB + wG;
	if (wSum > EPSILON) {
		pm = (pA + corrA + pB + corrB) * 0.5f;
		p2pm = pG + corrC - pm;
		float p2pm_mag = p2pm.length();
		p2pm *= 1.0f / p2pm_mag;
		lambda = (p2pm_mag - ghostEdgeRestLength) / wSum * edgeKs;
		corrA += p2pm * (wA * lambda *  .5f);
		corrB += p2pm * (wB * lambda *  .5f);
		corrC -= p2pm * (wG * lambda);
	}
	return true;
}

void ElasticRodEdgeConstraint::setEdgeKs(const float& x)
{ m_edgeKs = x; }
    
}
}

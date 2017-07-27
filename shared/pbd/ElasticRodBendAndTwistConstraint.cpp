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
    Vector3F* ps = part->projectedPos();
    Vector3F* gs = ghost->projectedPos();
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
	//return true;
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
    Vector3F darboux;
	Matrix33F dA, dB;

	computeMaterialFrame(dA, pA, pB, pD);
	computeMaterialFrame(dB, pB, pC, pE);
	computeDarbouxVector(darboux, dA, dB, midEdgeLength);
	
	Matrix33F dajpi[3][3];
	computeMaterialFrameDerivative(pA, pB, pD, dA,
		dajpi[0][0], dajpi[0][1], dajpi[0][2],
		dajpi[1][0], dajpi[1][1], dajpi[1][2],
		dajpi[2][0], dajpi[2][1], dajpi[2][2]);

	Matrix33F  dbjpi[3][3];
	computeMaterialFrameDerivative(pB, pC, pE, dB,
		dbjpi[0][0], dbjpi[0][1], dbjpi[0][2],
		dbjpi[1][0], dbjpi[1][1], dbjpi[1][2],
		dbjpi[2][0], dbjpi[2][1], dbjpi[2][2]);
	
	Matrix33F constraint_jacobian[5];
	computeDarbouxGradient(darboux, midEdgeLength, dA, dB, 
		dajpi, dbjpi, 
		bendingAndTwistingKs,
		constraint_jacobian[0],
		constraint_jacobian[1],
		constraint_jacobian[2],
		constraint_jacobian[3],
		constraint_jacobian[4]);
	
	Vector3F constraint_value(bendingAndTwistingKs.comp(0) * (darboux.comp(0) - restDarbouxVector.comp(0)),
                             bendingAndTwistingKs.comp(1) * (darboux.comp(1) - restDarbouxVector.comp(1)),
                             bendingAndTwistingKs.comp(2) * (darboux.comp(2) - restDarbouxVector.comp(2)) );

	Matrix33F factor_matrix;
	factor_matrix.setZero();
	
	Matrix33F tmp_mat;
	float invMasses[] = { wA, wB, wC, wD, wE };
	for (int i = 0; i < 5; ++i) {
		tmp_mat = constraint_jacobian[i] * constraint_jacobian[i];
		tmp_mat *= invMasses[i];

		factor_matrix += tmp_mat;
	}
	
	Vector3F dp[5];
	tmp_mat = factor_matrix;
	tmp_mat.inverse();

    for (int i = 0; i < 5; ++i) {
        constraint_jacobian[i] *= invMasses[i];
        dp[i] = constraint_jacobian[i] * (tmp_mat * constraint_value);
        dp[i] *= -1.f;
    }

    corrA = dp[0];
    corrB = dp[1];
    corrC = dp[2];
    corrD = dp[3];
    corrE = dp[4];
    return true;
}

///         p^g
///         |
///  pe_1 ----- pe
/// d^3e <- (pe - pe_1) / |(pe - pe_1)|  along the edge direction
/// d^2e <- d^3e x (p^ge - pe_1) / |d^3e x (p^ge - pe_1)| up
/// d^1e <- d^2e x d^3e front
/// De <- [d^1e, d^2e, d^3e]
/// material frame at the center of an edge
void ElasticRodBendAndTwistConstraint::computeMaterialFrame(Matrix33F& frame,
	        const Vector3F& vA, const Vector3F& vB, const Vector3F& vG)
{
    Vector3F d3 = vB - vA;
    d3.normalize();
    Vector3F d2 = d3.cross(vG - vA);
    d2.normalize();
    Vector3F d1 = d2.cross(d3);
    frame.fill(d1, d2, d3);
}

static const int permutation[3][3] = {
	{0, 2, 1},
	{1, 0, 2},
	{2, 1, 0}
};

/// Darboux vector describes how the material frame evolves along the curve
/// an axial vector of frame rotation with respect to change of s.
/// s is a point on rod
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

bool ElasticRodBendAndTwistConstraint::computeMaterialFrameDerivative(const Vector3F& p0, const Vector3F& p1, const Vector3F& p2, 
            const Matrix33F& d,
            Matrix33F& d1p0, Matrix33F& d1p1, Matrix33F& d1p2,
            Matrix33F& d2p0, Matrix33F& d2p1, Matrix33F& d2p2,
            Matrix33F& d3p0, Matrix33F& d3p1, Matrix33F& d3p2)
{
    /// d3pi
	Vector3F p01 = p1 - p0;
	float length_p01 = p01.length();

	const Vector3F d3 = d.row(2);
	d3p0.setRow(0, d3 * d3.x);
	d3p0.setRow(1, d3 * d3.y);
	d3p0.setRow(2, d3 * d3.z);

	*d3p0.m(0, 0) -= 1.0f;
	*d3p0.m(1, 1) -= 1.0f;
	*d3p0.m(2, 2) -= 1.0f;

	d3p0 *= 1.f / length_p01;

	d3p1 = d3p0 * -1.f;

	d3p2.setZero();
	
	std::cout<<"\n d3dp0"<<d3p0
	<<"\n d3dp1"<<d3p1
	<<"\n d3dp2"<<d3p2;

	//// d2pi
	Vector3F p02 = p2 - p0;
	Vector3F p01_cross_p02 = p01.cross(p02);
	float length_cross = p01_cross_p02.length();

	Matrix33F mat;
	const Vector3F dr1 = d.row(1);
	mat.setRow(0, dr1 * dr1.x);
	mat.setRow(1, dr1 * dr1.y);
	mat.setRow(2, dr1 * dr1.z);

	*mat.m(0,0) -= 1.f;
	*mat.m(1,1) -= 1.f;
	*mat.m(2,2) -= 1.f;

	mat *= 1.f / length_cross;

	Matrix33F product_matrix;
	product_matrix.asCrossProductMatrix(p2 - p1);
	d2p0 = mat * product_matrix;

	product_matrix.asCrossProductMatrix(p0 - p2);
	d2p1 = mat * product_matrix;

	product_matrix.asCrossProductMatrix(p1 - p0);
	d2p2 = mat * product_matrix;
	
	std::cout<<"\n d2dp0"<<d2p0
	<<"\n d2dp1"<<d2p1
	<<"\n d2dp2"<<d2p2;

	//// d1pi
	Matrix33F product_mat_d3;
	Matrix33F product_mat_d2;
	Matrix33F m1, m2;

	product_mat_d3.asCrossProductMatrix(d.row(2));
	product_mat_d2.asCrossProductMatrix(d.row(1));

	d1p0 = product_mat_d2 * d3p0 - product_mat_d3 * d2p0;

	d1p1 = product_mat_d2 * d3p1 - product_mat_d3 * d2p1;

	d1p2 = product_mat_d3 * d2p2;
	d1p2 *= -1.f;
	
	std::cout<<"\n d1dp0"<<d1p0
	<<"\n d1dp1"<<d1p1
	<<"\n d1dp2"<<d1p2;
	return true;
}

bool ElasticRodBendAndTwistConstraint::computeDarbouxGradient(
	const Vector3F& darboux_vector, const float length,
	const Matrix33F& da, const Matrix33F& db,
	const Matrix33F dajpi[3][3], const Matrix33F dbjpi[3][3],
	const Vector3F& bendAndTwistKs,
	Matrix33F& omega_pa, Matrix33F& omega_pb, Matrix33F& omega_pc, Matrix33F& omega_pd, Matrix33F& omega_pe
	)
{
    float x = 1.0f + da.row(0).dot(db.row(0)) 
                    + da.row(1).dot(db.row(1)) 
                    + da.row(2).dot(db.row(2));
	x = 2.0f / (length * x);

	for (int c = 0; c < 3; ++c) {
		const int i = permutation[c][0];
		const int j = permutation[c][1];
		const int k = permutation[c][2];
		// pa
		{
			Vector3F term1(0,0,0);
			Vector3F term2(0,0,0);
			Vector3F tmp(0,0,0);
			// first term
			term1 = dajpi[j][0] * db.row(k);
			tmp =   dajpi[k][0] * db.row(j);
			term1 = term1 - tmp;
			// second term
			for (int n = 0; n < 3; ++n) {
				tmp = dajpi[n][0] * db.row(n);
				term2 = term2 + tmp;
			}
			
			tmp = term1 - term2 * (0.5f * darboux_vector.comp(i) * length);
			tmp *= x * bendAndTwistKs.comp(i);
			omega_pa.setRow(i, tmp);
			
		}
		// pb
		{
			Vector3F term1(0, 0, 0);
			Vector3F term2(0, 0, 0);
			Vector3F tmp(0, 0, 0);
			// first term
			term1 = dajpi[j][1] * db.row(k);
			tmp =   dajpi[k][1] * db.row(j);
			term1 = term1 - tmp;
			// third term
			tmp = dbjpi[j][0] * da.row(k);
			term1 = term1 - tmp;
			
			tmp = dbjpi[k][0] * da.row(j);
			term1 = term1 + tmp;

			// second term
			for (int n = 0; n < 3; ++n) {
				tmp = dajpi[n][1] * db.row(n);
				term2 = term2 + tmp;
				
				tmp = dbjpi[n][0] * da.row(n);
				term2 = term2 + tmp;
			}
			
			tmp = term1 - term2 *(0.5f * darboux_vector.comp(i) * length);
			tmp *= x * bendAndTwistKs.comp(i);
			omega_pb.setRow(i, tmp);
			
		}
		// pc
		{
			Vector3F term1(0, 0, 0);
			Vector3F term2(0, 0, 0);
			Vector3F tmp(0, 0, 0);
			
			// first term
			term1 = dbjpi[j][1] * da.row(k);
			tmp =   dbjpi[k][1] * da.row(j);
			term1 = term1 - tmp;

			// second term
			for (int n = 0; n < 3; ++n) {
				tmp = dbjpi[n][1] * da.row(n);
				term2 = term2 + tmp;
			}
			
			tmp = term1 + term2 * (0.5f * darboux_vector.comp(i) * length);
			tmp *= -x * bendAndTwistKs.comp(i);
			omega_pc.setRow(i, tmp);
			
		}
		// pd
		{
			Vector3F term1(0, 0, 0);
			Vector3F term2(0, 0, 0);
			Vector3F tmp(0, 0, 0);
			// first term
			term1 = dajpi[j][2] * db.row(k);
			tmp =   dajpi[k][2] * db.row(j);
			term1 = term1 - tmp;
			// second term
			for (int n = 0; n < 3; ++n) {
				tmp = dajpi[n][2] * db.row(n);
				term2 = term2 + tmp;
			}
			tmp = term1 - term2 * (0.5f * darboux_vector.comp(i) * length);
			tmp *= x * bendAndTwistKs.comp(i);
			omega_pd.setRow(i, tmp);
		}
		// pe
		{
			Vector3F term1(0, 0, 0);
			Vector3F term2(0, 0, 0);
			Vector3F tmp(0, 0, 0);
			// first term
			term1 = dbjpi[j][2] * da.row(k);
			tmp = dbjpi[k][2] * da.row(j);
			term1 -= tmp;
			
			// second term
			for (int n = 0; n < 3; ++n) {	
			    tmp = dbjpi[n][2] * da.row(n);
				term2 += tmp;
			}

			tmp = term1 + term2 * (0.5f * darboux_vector.comp(i) * length);
			tmp *= -x * bendAndTwistKs.comp(i);
			omega_pe.setRow(i, tmp);
		}
	}
	return true;
}

}
}

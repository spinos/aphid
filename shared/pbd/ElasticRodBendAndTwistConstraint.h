/*
 *  five point constraint for each edge pair
 *
 *     D     E  ghost point for frame representation
 *     |     |
 *  A --- B --- C
 *
 */
#ifndef APH_PBD_ELASTIC_ROD_BEND_TWIST_CONSTRAINT_H
#define APH_PBD_ELASTIC_ROD_BEND_TWIST_CONSTRAINT_H

#include "Constraint.h"
#include <math/MatrixC33F.h>

namespace aphid {

namespace pbd {

class ParticleData;

class ElasticRodBendAndTwistConstraint : public Constraint<5> {
 
	Vector3F m_restDarbouxVector;
	Vector3F m_bendAndTwistKs;
/// |D - edgeAB midpoint|
	float m_midEdgeRestLength;
	float m_stiffness;
	
public:
    ElasticRodBendAndTwistConstraint();
    
    virtual ConstraintType getConstraintType() const;
    
    bool initConstraint(SimulationContext * model, const int pA, const int pB, const int pC,
                                    const int pD, const int pE);
	void updateConstraint(SimulationContext * model);
	
	void setStiffness(float x);
	bool solvePositionConstraint(ParticleData* part, ParticleData* ghost);
	void setBendAndTwistKs(const float& a, const float& b, const float& c);
	
	void calculateGeometryNormal(ParticleData* part, ParticleData* ghost);

	const Vector3F& bendAndTwistKs() const;
	const float& midEdgeRestLength() const;
	const float& stiffness() const;

protected:

	Vector3F& restDarbouxVector();
	void setEdgeRestLength(float x);
	bool projectBendingAndTwistingConstraint(const Vector3F& pA, const float wA, 
		const Vector3F& pB, const float wB, 
		const Vector3F& pC, const float wC,
		const Vector3F& pD, const float wD,
		const Vector3F& pE, const float wE,
		const Vector3F& bendingAndTwistingKs,
		const float midEdgeLength,
		const Vector3F& restDarbouxVector,
		Vector3F& corrA, Vector3F& corrB, Vector3F& corrC, Vector3F& corrD, Vector3F& corrE) const;
	void computeMaterialFrame(MatrixC33F& frame,
	        const Vector3F& vA, const Vector3F& vB, const Vector3F& vG) const;
	void computeDarbouxVector(Vector3F& darboux,
	        const MatrixC33F& frameA, const MatrixC33F& frameB,
	        float midEdgeLength) const;
	bool computeMaterialFrameDerivative(const Vector3F& p0, const Vector3F& p1, const Vector3F& p2, 
            const MatrixC33F& d,
            MatrixC33F& d1p0, MatrixC33F& d1p1, MatrixC33F& d1p2,
            MatrixC33F& d2p0, MatrixC33F& d2p1, MatrixC33F& d2p2,
            MatrixC33F& d3p0, MatrixC33F& d3p1, MatrixC33F& d3p2) const;
    bool computeDarbouxGradient(const Vector3F& darboux_vector, 
            const float length,
            const MatrixC33F& da, const MatrixC33F& db,
            const MatrixC33F dajpi[3][3], const MatrixC33F dbjpi[3][3],
            const Vector3F& bendAndTwistKs,
            MatrixC33F& omega_pa, MatrixC33F& omega_pb, MatrixC33F& omega_pc, 
			MatrixC33F& omega_pd, MatrixC33F& omega_pe) const;
	const Vector3F& restDarbouxVector() const;
	
};

}
}
#endif        //  #ifndef APH_PBD_ELASTIC_ROD_BEND_TWIST_CONSTRAINT_H


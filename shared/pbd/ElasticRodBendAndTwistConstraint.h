/*
 *  five point constraint for each edge pair
 *
 *     D     E  ghost point for frame representation
 *
 *  A --- B --- C
 *
 */
#ifndef APH_PBD_ELASTIC_ROD_BEND_TWIST_CONSTRAINT_H
#define APH_PBD_ELASTIC_ROD_BEND_TWIST_CONSTRAINT_H

#include "Constraint.h"
#include <math/Matrix33F.h>

namespace aphid {

namespace pbd {

class ParticleData;

class ElasticRodBendAndTwistConstraint : public Constraint<5> {

    Vector3F m_bendAndTwistKs;
	Vector3F m_restDarbouxVector;
	Matrix33F m_dA; //material frame A
	Matrix33F m_dB; //material frame B
		
public:
    ElasticRodBendAndTwistConstraint();
    
    virtual ConstraintType getConstraintType() const;
    
    bool initConstraint(SimulationContext * model, const int pA, const int pB, const int pC,
                                    const int pD, const int pE);
	bool solvePositionConstraint(ParticleData* part, ParticleData* ghost);
	
private:
	void computeMaterialFrame(Matrix33F& frame,
	        const Vector3F& vA, const Vector3F& vB, const Vector3F& vG);
	void computeDarbouxVector(Vector3F& darboux,
	        const Matrix33F& frameA, const Matrix33F& frameB,
	        float midEdgeLength);
	bool projectBendingAndTwistingConstraint(const Vector3F& pA, const float wA, 
		const Vector3F& pB, const float wB, 
		const Vector3F& pC, const float wC,
		const Vector3F& pD, const float wD,
		const Vector3F& pE, const float wE,
		const Vector3F& bendingAndTwistingKs,
		const float midEdgeLength,
		const Vector3F& restDarbouxVector,
		Vector3F& corrA, Vector3F& corrB, Vector3F& corrC, Vector3F& corrD, Vector3F& corrE);
	bool computeMaterialFrameDerivative(const Vector3F& p0, const Vector3F& p1, const Vector3F& p2, 
            const Matrix33F& d,
            Matrix33F& d1p0, Matrix33F& d1p1, Matrix33F& d1p2,
            Matrix33F& d2p0, Matrix33F& d2p1, Matrix33F& d2p2,
            Matrix33F& d3p0, Matrix33F& d3p1, Matrix33F& d3p2);
    bool computeDarbouxGradient(const Vector3F& darboux_vector, 
            const float length,
            const Matrix33F& da, const Matrix33F& db,
            const Matrix33F dajpi[3][3], const Matrix33F dbjpi[3][3],
            const Vector3F& bendAndTwistKs,
            Matrix33F& omega_pa, Matrix33F& omega_pb, Matrix33F& omega_pc, Matrix33F& omega_pd, Matrix33F& omega_pe
            );
	
};

}
}
#endif        //  #ifndef APH_PBD_ELASTIC_ROD_BEND_TWIST_CONSTRAINT_H


/*
 *  three point constraint for each edge
 *
 *     G  ghost point for frame representation
 *
 *  A --- B
 *
 */
#ifndef APH_PBD_ELASTIC_ROD_EDGE_CONSTRAINT_H
#define APH_PBD_ELASTIC_ROD_EDGE_CONSTRAINT_H

#include "Constraint.h"

namespace aphid {

class Vector3F;

namespace pbd {

class ParticleData;

class ElasticRodEdgeConstraint : public Constraint<3> {

    float m_restLength;
    float m_ghostRestLength;
    float m_edgeKs;
    
public:
    ElasticRodEdgeConstraint();
    
    virtual ConstraintType getConstraintType() const;
    
    bool initConstraint(SimulationContext * model, const int pA, const int pB, const int pG);
	bool solvePositionConstraint(ParticleData* part, ParticleData* ghost);
	
	void setEdgeKs(const float& x);
	
private:
	bool projectEdgeConstraints(const Vector3F& pA, const float wA, 
		const Vector3F& pB, const float wB, 
		const Vector3F& pG, const float wG,
		const float edgeKs, const float edgeRestLength, const float ghostEdgeRestLength,
		Vector3F& corrA, Vector3F& corrB, Vector3F& corrC);
		
};

}
}
#endif        //  #ifndef APH_PBD_ELASTIC_ROD_EDGE_CONSTRAINT_H


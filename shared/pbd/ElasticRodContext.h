/*
 *  Position-Based-Dynamics
 *  Discrete Rod with ghost particles
 */
#ifndef APH_PBD_ELASTIC_ROD_CONTEXT_H
#define APH_PBD_ELASTIC_ROD_CONTEXT_H

#include "SimulationContext.h"

namespace aphid {
namespace pbd {
class ElasticRodEdgeConstraint;
class ElasticRodContext : public SimulationContext {

    ParticleData m_ghostPart;
    typedef std::vector<ElasticRodEdgeConstraint* > EdgeConstraintVector;
    EdgeConstraintVector m_edgeConstraints;
    
public:
    ElasticRodContext();
    virtual ~ElasticRodContext();
    
    const ParticleData* c_ghostParticles() const;
    
protected:
    ParticleData* ghostParticles();
    void addElasticRodEdgeConstraint(int a, int b, int g);
    virtual void positionConstraintProjection();
	
private:
    void clearConstraints();
    
};

}
}
#endif        //  #ifndef ELASTICRODCONTEXT_H


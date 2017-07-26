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
/// velocity of mid-point of each edge
    Vector3F * m_vmt;
    
public:
    ElasticRodContext();
    virtual ~ElasticRodContext();
    
    virtual const ParticleData* c_ghostParticles() const;
    
protected:
    ParticleData* ghostParticles();
    void addElasticRodEdgeConstraint(int a, int b, int g);
    void addElasticRodBendAndTwistConstraint(int a, int b, int c,
                                    int d, int e);
    void positionConstraintProjection();
/// modify gravity on ghost points
	virtual void applyGravity(float dt);
	
	int numEdges() const;
/// by num edge constraints
	void createEdges();
	
	virtual void dampVelocity(float damping);
	virtual void projectPosition(float dt);
    virtual void updateVelocityAndPosition(float dt);
    
private:
    void clearConstraints();
    void modifyEdgeGravity(Vector3F& vA, Vector3F& vB, Vector3F& vG,
                        Vector3F& vmt_1, float dt);
    
};

}
}
#endif        //  #ifndef ELASTICRODCONTEXT_H


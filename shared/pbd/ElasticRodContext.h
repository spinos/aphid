/*
 *  Position-Based-Dynamics
 *  Discrete Rod with ghost particles
 */
#ifndef APH_PBD_ELASTIC_ROD_CONTEXT_H
#define APH_PBD_ELASTIC_ROD_CONTEXT_H

#include "SimulationContext.h"

namespace aphid {
namespace pbd {
class Beam;
class ElasticRodEdgeConstraint;
class ElasticRodBendAndTwistConstraint;
class ElasticRodContext : public SimulationContext {

    ParticleData m_ghostPart;
    typedef std::vector<ElasticRodEdgeConstraint* > EdgeConstraintVector;
    EdgeConstraintVector m_edgeConstraints;
    typedef std::vector<ElasticRodBendAndTwistConstraint* > BendTwistConstraintVector;
    BendTwistConstraintVector m_bendTwistConstraints;
/// velocity of mid-point of each edge
    Vector3F * m_vmt;
    
public:
    ElasticRodContext();
    virtual ~ElasticRodContext();
    
    virtual ParticleData* ghostParticles();
    virtual const ParticleData* c_ghostParticles() const;
    virtual void applyWind(float dt);
    int numEdges() const;
    void getEdgeIndices(int& iA, int& iB, int& iG, 
                        const int& i) const;
    
protected:
/// create particles from beam models
	void createBeams(const Beam* bems, int numBeams);
    void addElasticRodEdgeConstraint(int a, int b, int g);
    void addElasticRodBendAndTwistConstraint(int a, int b, int c,
                                    int d, int e, float stiffness);
    void positionConstraintProjection();
/// modify gravity on ghost points
	virtual void applyGravity(float dt);
	
/// by num edge constraints
	void createEdges();
	
	virtual void dampVelocity(float damping);
	virtual void projectPosition(float dt);
    virtual void updateVelocityAndPosition(float dt);
	
	void addElasticRodBendAndTwistConstraint(ElasticRodBendAndTwistConstraint* c);
	
	ElasticRodBendAndTwistConstraint* bendAndTwistConstraint(int i);
    
	void computeGeometryNormal();
	
private:
    void clearConstraints();
    void modifyEdgeGravity(Vector3F& vA, Vector3F& vB, Vector3F& vG,
                        Vector3F& vmt_1, float dt);
    
};

}
}
#endif        //  #ifndef ELASTICRODCONTEXT_H


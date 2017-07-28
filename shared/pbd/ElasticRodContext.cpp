#include "ElasticRodContext.h"
#include "ElasticRodEdgeConstraint.h"
#include "ElasticRodBendAndTwistConstraint.h"

namespace aphid {
namespace pbd {

ElasticRodContext::ElasticRodContext() :
m_vmt(0)
{}

ElasticRodContext::~ElasticRodContext()
{ clearConstraints(); }

ParticleData* ElasticRodContext::ghostParticles()
{ return &m_ghostPart; }

const ParticleData* ElasticRodContext::c_ghostParticles() const
{ return &m_ghostPart; }

void ElasticRodContext::addElasticRodEdgeConstraint(int a, int b, int g)
{
    ElasticRodEdgeConstraint *c = new ElasticRodEdgeConstraint();
    const bool res = c->initConstraint(this, a, b, g);
    if (res)
        m_edgeConstraints.push_back(c);
}

void ElasticRodContext::addElasticRodBendAndTwistConstraint(int a, int b, int c,
                                    int d, int e)
{   
    ElasticRodBendAndTwistConstraint *cn = new ElasticRodBendAndTwistConstraint();
    const bool res = cn->initConstraint(this, a, b, c, d, e);
    if (res) 
        m_bendTwistConstraints.push_back(cn);
}

void ElasticRodContext::addElasticRodBendAndTwistConstraint(ElasticRodBendAndTwistConstraint* c)
{ m_bendTwistConstraints.push_back(c); }

void ElasticRodContext::clearConstraints()
{
    EdgeConstraintVector::iterator it = m_edgeConstraints.begin();
    for(;it!=m_edgeConstraints.end();++it) {
        delete *it;
    }
    m_edgeConstraints.clear();
    
    BendTwistConstraintVector::iterator itbt = m_bendTwistConstraints.begin();
    for(;itbt!=m_bendTwistConstraints.end();++itbt) {
        delete *itbt;
    }
    m_bendTwistConstraints.clear();
}

void ElasticRodContext::positionConstraintProjection()
{
	const int nec = m_edgeConstraints.size();
	const int nbtc = m_bendTwistConstraints.size();
	int nloop = 0;
	while(nloop < 5) {
#if 1
		EdgeConstraintVector::iterator it = m_edgeConstraints.begin();
		for(;it!=m_edgeConstraints.end();++it) {
			(*it)->solvePositionConstraint(particles(), ghostParticles() );
		}
		
		BendTwistConstraintVector::iterator itb = m_bendTwistConstraints.begin();
		for(;itb!=m_bendTwistConstraints.end();++itb) {
			(*itb)->solvePositionConstraint(particles(), ghostParticles() );
		}
#else
		for(int i=0;i<nec;++i) {
			m_edgeConstraints[i]->solvePositionConstraint(particles(), ghostParticles() );
			if(i<nbtc) {
				m_bendTwistConstraints[i]->solvePositionConstraint(particles(), ghostParticles() );
			}
		}
#endif		
		nloop++;
	}
}

int ElasticRodContext::numEdges() const
{ return m_edgeConstraints.size(); }

void ElasticRodContext::applyGravity(float dt)
{
    SimulationContext::applyGravity(dt);
/// gravity on ghost points
    const int& ng = ghostParticles()->numParticles();
	Vector3F* velg = ghostParticles()->velocity();
	const float* im = ghostParticles()->invMass();
	for(int i=0;i< ng;i++) {
	    if(im[i] > 0.f) velg[i].y -= 9.8f * dt;
	}
	
    const int ne = numEdges();
    Vector3F* vel = particles()->velocity();
    
    for(int i=0;i<ne;++i) {
        const ElasticRodEdgeConstraint * c = m_edgeConstraints[i];
        const int vA = c->c_bodyInds()[0];
        const int vB = c->c_bodyInds()[1];
        const int vG = c->c_bodyInds()[2];
        modifyEdgeGravity(vel[vA], vel[vB], velg[vG], m_vmt[i], dt);
    }
}

void ElasticRodContext::projectPosition(float dt)
{
    SimulationContext::projectPosition(dt);
    m_ghostPart.projectPosition(dt);
}

void ElasticRodContext::createEdges()
{
    const int ne = numEdges();
    m_vmt = new Vector3F[ne];
    memset(m_vmt, 0, 12 * ne);
}

void ElasticRodContext::modifyEdgeGravity(Vector3F& vA, Vector3F& vB, Vector3F& vG,
                                Vector3F& vmt_1, float dt)
{
/// the velocity of the mid-point at time t
    Vector3F vmt = (vA + vB) * .5f;
/// acceleration of the mid-point
    Vector3F am = (vmt - vmt_1) / dt;
/// update vmt
    vmt_1 = vmt;
    
    static const Vector3F gv(0.f, -9.8, 0.f);
    
/// r <- (am .g) / |g|2
    float r = 1.f - am.dot(gv) / 96.04f;

    vG -= gv * r * dt;
    vA += gv * 0.5 * r * dt;
    vB += gv * 0.5 * r * dt;
}

void ElasticRodContext::updateVelocityAndPosition(float dt)
{
    SimulationContext::updateVelocityAndPosition(dt);
    m_ghostPart.updateVelocityAndPosition(dt);
}

void ElasticRodContext::dampVelocity(float damping)
{ 
    SimulationContext::dampVelocity(damping);
    m_ghostPart.dampVelocity(damping); 
} 

ElasticRodBendAndTwistConstraint* ElasticRodContext::bendAndTwistConstraint(int i)
{ return m_bendTwistConstraints[i]; }

void ElasticRodContext::getEdgeIndices(int& iA, int& iB, int& iG,
                                    const int& i) const
{
    ElasticRodEdgeConstraint *c = m_edgeConstraints[i];
    iA = c->c_bodyInds()[0];
    iB = c->c_bodyInds()[1];
    iG = c->c_bodyInds()[2];
}

}
}

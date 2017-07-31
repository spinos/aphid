#include "ElasticRodContext.h"
#include "ElasticRodEdgeConstraint.h"
#include "ElasticRodBendAndTwistConstraint.h"
#include "Beam.h"

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
                                    int d, int e, float stiffness)
{   
    ElasticRodBendAndTwistConstraint *cn = new ElasticRodBendAndTwistConstraint();
    const bool res = cn->initConstraint(this, a, b, c, d, e);
    if (res) {
		cn->setStiffness(stiffness);
        m_bendTwistConstraints.push_back(cn);
	}
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
	while(nloop < 4) {
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
	applyGravityTo(ghostParticles(), dt);
}

void ElasticRodContext::modifyGhostGravity(float dt)
{    
	Vector3F* velg = ghostParticles()->velocity();
	
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
    
	const Vector3F gv = getGravityVec();
	
/// r <- (am . g) / |g|2
    float r = 1.f - am.dot(gv ) / (grivityY() * grivityY() );

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

void ElasticRodContext::createBeams(const Beam* bems, int numBeams)
{
	int np = 0;
	int ngp = 0;
	
	for(int j=0;j<numBeams;++j) {
		np += bems[j].numParticles();
		ngp += bems[j].numGhostParticles();
	
	}
	
    pbd::ParticleData* part = particles();
	part->createNParticles(np);
	pbd::ParticleData* ghost = ghostParticles();
	ghost->createNParticles(ngp);
	
	int npbegin = 0;
	int ngpbegin = 0;
	for(int j=0;j<numBeams;++j) {
		const int npj = bems[j].numParticles();
		const int ngpj = bems[j].numGhostParticles();
		
		for(int i=0;i<npj;++i) {
			part->setParticle(bems[j].getParticlePnt(i), npbegin + i);
			part->invMass()[npbegin + i] = bems[j].getInvMass(i);
		}
    
		for(int i=0;i<ngpj;++i) {
			ghost->setParticle(bems[j].getGhostParticlePnt(i), ngpbegin + i);
			ghost->invMass()[ngpbegin + i] = bems[j].getInvMass(i);
		}
    
///lock two first particles and first ghost point
		part->invMass()[npbegin + 0] = 0.f;
		part->invMass()[npbegin + 1] = 0.f;
		ghost->invMass()[ngpbegin + 0] = 0.f;
    
		const int& nsj = bems[j].numSegments();
		//std::cout<<"\n nseg "<<nsj;
		for(int i=0;i<nsj;++i) {
			const int& ci = bems[j].getConstraintSegInd(i);
			addElasticRodEdgeConstraint(ci + npbegin, ci+1 + npbegin, ci + ngpbegin);
			//std::cout<<"\n eg "<<ci<<" "<<(ci+1)<<" "<<ci;
		}
		
		for(int i=0;i<nsj;++i) {
			const int& ci = bems[j].getConstraintSegInd(i);
			if(ci < nsj - 1) {
				addElasticRodBendAndTwistConstraint(ci + npbegin, ci+1 + npbegin, ci+2 + npbegin, 
											ci + ngpbegin, ci+1 + ngpbegin,
											bems[j].getStiffness(i+1) );
			}
		}
		
		npbegin += npj;
		ngpbegin += ngpj;
		
	}
	
}

void ElasticRodContext::computeGeometryNormal()
{
	BendTwistConstraintVector::iterator itb = m_bendTwistConstraints.begin();
	for(;itb!=m_bendTwistConstraints.end();++itb) {
		(*itb)->calculateGeometryNormal(particles(), ghostParticles() );
	}
}

void ElasticRodContext::applyWind(float dt)
{
	computeGeometryNormal();
	SimulationContext::applyWind(dt);
	applyWindTo(&m_ghostPart, dt);
}

}
}

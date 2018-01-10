#include "ElasticRodContext.h"
#include "ElasticRodEdgeConstraint.h"
#include "ElasticRodBendAndTwistConstraint.h"
#include "ElasticRodAttachmentConstraint.h"
#include "Beam.h"
#include <geom/ParallelTransport.h>
#include <math/Matrix33F.h>

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

void ElasticRodContext::addElasticRodAttachmentConstraint(int a, int b, int c,
                                    int d, int e, float stiffness)
{
	ElasticRodAttachmentConstraint* ac = new ElasticRodAttachmentConstraint();
	ac->initConstraint(this, a, b, c, d, e);
	ac->setStiffness(stiffness);
	m_attachmentConstraints.push_back(ac);
}

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
	while(nloop < 3) {
#if 1
		EdgeConstraintVector::iterator it = m_edgeConstraints.begin();
		for(;it!=m_edgeConstraints.end();++it) {
			(*it)->solvePositionConstraint(particles(), ghostParticles() );
		}
		
		BendTwistConstraintVector::iterator itb = m_bendTwistConstraints.begin();
		for(;itb!=m_bendTwistConstraints.end();++itb) {
			(*itb)->solvePositionConstraint(particles(), ghostParticles() );
		}
		
		AttachmentConstraintVector::iterator ita = m_attachmentConstraints.begin();
		for(;ita!=m_attachmentConstraints.end();++ita) {
			(*ita)->solvePositionConstraint(particles(), ghostParticles() );
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
	//applyGravityTo(ghostParticles(), dt);
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
	m_numStrands = numBeams;
	m_strandBegin = new int[m_numStrands + 1];
	m_strandGhostBegin = new int[m_numStrands + 1];
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
	
	int npj, ngpj;
	int npbegin = 0;
	int ngpbegin = 0;
	for(int j=0;j<numBeams;++j) {
		npj = bems[j].numParticles();
		ngpj = bems[j].numGhostParticles();
		
		for(int i=0;i<npj;++i) {
			part->setParticle(bems[j].getParticlePnt(i), npbegin + i);
			part->invMass()[npbegin + i] = bems[j].getInvMass(i);
		}
    
		for(int i=0;i<ngpj;++i) {
			ghost->setParticle(bems[j].getGhostParticlePnt(i), ngpbegin + i);
			ghost->invMass()[ngpbegin + i] = bems[j].getInvMass(i);
		}
    
///lock first particles and first ghost point
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
#if 0		
		addElasticRodAttachmentConstraint(npbegin, 1 + npbegin, 1 + npbegin, 
											ngpbegin, ngpbegin,
											.5f );
#endif
		m_strandBegin[j] = npbegin;
		m_strandGhostBegin[j] = ngpbegin;
	
		npbegin += npj;
		ngpbegin += ngpj;
	}
	m_strandBegin[m_numStrands] = npbegin;
	m_strandGhostBegin[m_numStrands] = ngpj;
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
	//computeGeometryNormal();
	SimulationContext::applyWind(dt);
	//applyWindTo(&m_ghostPart, dt);
}

void ElasticRodContext::resetBendAndTwistConstraints()
{
	for(int i=0;i<m_numStrands;++i) {
		resetGhostPosition(m_strandBegin[i], m_strandBegin[i + 1],
			m_strandGhostBegin[i], m_strandGhostBegin[i + 1]);
	}
	
	BendTwistConstraintVector::iterator itb = m_bendTwistConstraints.begin();
	for(;itb!=m_bendTwistConstraints.end();++itb) {
		(*itb)->updateConstraint(this);
	}
}

void ElasticRodContext::resetGhostPosition(const int& pbegin, const int& pend,
				const int& gbegin, const int& gend)
{
	pbd::ParticleData* part = particles();
	pbd::ParticleData* ghost = ghostParticles();
	const Vector3F* ps = &part->pos()[pbegin];
	Vector3F* gs = &ghost->pos()[gbegin];
	const int np = pend - pbegin;
	Matrix33F frm;
	Vector3F p0p1 = ps[1] - ps[0];
	ParallelTransport::FirstFrame(frm, p0p1, Vector3F(0.f, 1.f, 0.f) );
	gs[0] = (ps[0] + ps[1]) * .5f + ParallelTransport::FrameUp(frm) * p0p1.length();
	
	Vector3F p1p2;
	for(int i=1;i<np-1;++i) {
		const Vector3F& p1 = ps[i];
		const Vector3F& p2 = ps[i+1];
		p1p2 = p2 - p1;
		ParallelTransport::RotateFrame(frm, p0p1, p1p2);
		
		gs[i] = (p1 + p2) * .5f + ParallelTransport::FrameUp(frm) * (p1.distanceTo(p2) * .5f);
		
		p0p1 = p1p2;
	}
}

}
}

#include "ElasticRodContext.h"
#include "ElasticRodEdgeConstraint.h"

namespace aphid {
namespace pbd {

ElasticRodContext::ElasticRodContext()
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

void ElasticRodContext::clearConstraints()
{
    EdgeConstraintVector::iterator it = m_edgeConstraints.begin();
    for(;it!=m_edgeConstraints.end();++it) {
        delete *it;
    }
    m_edgeConstraints.clear();
}

void ElasticRodContext::positionConstraintProjection()
{
	int nloop = 0;
	while(nloop < 4) {
		EdgeConstraintVector::iterator it = m_edgeConstraints.begin();
		for(;it!=m_edgeConstraints.end();++it) {
			(*it)->solvePositionConstraint(particles(), ghostParticles() );
			nloop++;
		}
	}
}

}
}

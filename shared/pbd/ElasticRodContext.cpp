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

}
}

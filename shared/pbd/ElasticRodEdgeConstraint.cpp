#include "ElasticRodEdgeConstraint.h"
#include "SimulationContext.h"

namespace aphid {
namespace pbd {
    
ElasticRodEdgeConstraint::ElasticRodEdgeConstraint()
{}

ElasticRodEdgeConstraint::ConstraintType ElasticRodEdgeConstraint::getConstraintType() const
{ return ctElasticRodEdge; }

bool ElasticRodEdgeConstraint::initConstraint(SimulationContext * model, const int pA, const int pB, const int pG)
{
    bodyInds()[0] = pA;
    bodyInds()[1] = pB;
    bodyInds()[2] = pG;
    
    const Vector3F* ps = model->c_particles()->pos();
    m_restLength = ps[pA].distanceTo(ps[pB]);
    return true;
}
    
}
}

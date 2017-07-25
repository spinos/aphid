#ifndef APH_PBD_ELASTIC_ROD_EDGE_CONSTRAINT_H
#define APH_PBD_ELASTIC_ROD_EDGE_CONSTRAINT_H

#include "Constraint.h"

namespace aphid {
namespace pbd {

class ElasticRodEdgeConstraint : public Constraint<3> {

    float m_restLength;
    
public:
    ElasticRodEdgeConstraint();
    
    virtual ConstraintType getConstraintType() const;
    
    bool initConstraint(SimulationContext * model, const int pA, const int pB, const int pG);
	
};

}
}
#endif        //  #ifndef APH_PBD_ELASTIC_ROD_EDGE_CONSTRAINT_H


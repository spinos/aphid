#ifndef APH_PBD_CONSTRAINT_H
#define APH_PBD_CONSTRAINT_H

namespace aphid {
namespace pbd {
    
class SimulationContext;
   
template<int NumBodies>
class Constraint {

    int m_bodyInds[NumBodies];
    
public:
    enum ConstraintType {
        ctUnknown = 0,
        ctElasticRodEdge = 1,
        ctElasticRodBendAndTwist = 2
    };
    
    Constraint();
   
    const int numBodies() const;
    
    virtual ConstraintType getConstraintType() const
    { return ctUnknown; }
    
    int* bodyInds();
    const int* c_bodyInds() const;
    
};

template<int NumBodies>
Constraint<NumBodies>::Constraint()
{}

template<int NumBodies>
const int Constraint<NumBodies>::numBodies() const
{ return NumBodies; }

template<int NumBodies>
int* Constraint<NumBodies>::bodyInds()
{ return m_bodyInds; }

template<int NumBodies>
const int* Constraint<NumBodies>::c_bodyInds() const
{ return m_bodyInds; }

}
}
#endif        //  #ifndef APH_PBD_CONSTRAINT_H


#include "contactConstraint.cuh"

namespace contactconstraint {
    
    void prepareNoPenetratingContact(ContactConstraint* constraints,
    float3 * contactLinearVel,
                                        uint2 * splits,
                                        uint2 * pairs,
                                        float3 * pos,
                                        float3 * vel,
                                        float3 * impulse,
                                        float * splitMass,
                                        ContactData * contacts,
                                        uint4 * tetind,
                                        uint numContacts2)
{
    dim3 block(SETCONSTRAINT_TPB, 1, 1);
    unsigned nblk = iDivUp(numContacts2, SETCONSTRAINT_TPB);
    dim3 grid(nblk, 1, 1);
    
    prepareNoPenetratingContactConstraint_kernel<<< grid, block >>>(constraints,
        contactLinearVel,
                                        splits,
                                        pos,
                                        vel,
                                        impulse,
                                        splitMass,
                                        contacts,
                                        tetind,
                                        numContacts2);
}

}

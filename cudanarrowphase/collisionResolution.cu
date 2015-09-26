#include "collisionResolution.cuh"
namespace collisionres {


void resolveCollision(ContactConstraint* constraints,
                        float3 * contactLinearVelocity,
                        float3 * deltaLinearVelocity,
	                    uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    uint numContacts2)
{
    dim3 block(SOLVECONTACT_TPB, 1, 1);
    unsigned nblk = iDivUp(numContacts2, SOLVECONTACT_TPB);
    dim3 grid(nblk, 1, 1);
    
    resolveCollision_kernel<<< grid, block >>>(constraints,
                        contactLinearVelocity,
                        deltaLinearVelocity,
	                    pairs,
                        splits,
	                    splitMass,
	                    contacts,
	                    numContacts2);
}

void resolveFriction(ContactConstraint* constraints,
                        float3 * contactLinearVelocity,
                        float3 * deltaLinearVelocity,
	                    uint2 * pairs,
                        uint2 * splits,
	                    float * splitMass,
	                    ContactData * contacts,
	                    uint numContacts2)
{
    dim3 block(SOLVECONTACT_TPB, 1, 1);
    unsigned nblk = iDivUp(numContacts2, SOLVECONTACT_TPB);
    dim3 grid(nblk, 1, 1);
    
    resolveFriction_kernel<<< grid, block >>>(constraints,
                        contactLinearVelocity,
                        deltaLinearVelocity,
	                    pairs,
                        splits,
	                    splitMass,
	                    contacts,
	                    numContacts2);
}

}

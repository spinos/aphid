#include "RaceCircuit.h"
#include "Silverstone.h"
#include <DynamicsSolver.h>
#include "PhysicsState.h"
namespace caterpillar {
RaceCircuit::RaceCircuit() {}
RaceCircuit::~RaceCircuit() {}
void RaceCircuit::create()
{
    btVector3 * pos = createVertexPos(sNumVertices);
    int i;
    for(i = 0; i < sNumVertices; i++) {
        pos[i] = btVector3(sMeshVertices[i * 3], sMeshVertices[i * 3 + 1], sMeshVertices[i * 3 + 2]);
    }
	int * idx = createTriangles(sNumTriangleIndices / 3);
	for(i = 0; i < sNumTriangleIndices; i++) {
        idx[i] = sMeshTriangleIndices[i];
    }
    
    setMargin(2.f);
	btBvhTriangleMeshShape* shp = createCollisionShape();
	
	btTransform trans; trans.setIdentity();
	btRigidBody * bd = PhysicsState::engine->createRigidBody(shp, trans, 0.f);
	bd->setFriction(.768f);
}
}

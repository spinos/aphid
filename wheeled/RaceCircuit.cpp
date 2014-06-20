#include "RaceCircuit.h"
#include "Silverstone.h"
#include <DynamicsSolver.h>
#include "PhysicsState.h"
namespace caterpillar {
RaceCircuit::RaceCircuit() {}
RaceCircuit::~RaceCircuit() {}
void RaceCircuit::create()
{
    setMargin(2.f);
    
    const float scaling = PhysicsState::engine->simulateScale();
	
    btVector3 * pos = createVertexPos(sNumVertices);
    int i;
    for(i = 0; i < sNumVertices; i++) {
        Vector3F q(sMeshVertices[i * 3] - sMeshNormals[i * 3] * margin(), sMeshVertices[i * 3 + 1]  - sMeshNormals[i * 3 + 1] * margin(), sMeshVertices[i * 3 + 2]  - sMeshNormals[i * 3 + 2] * margin());
        q *= scaling;
        pos[i] = btVector3(q.x, q.y, q.z);
    }
	int * idx = createTriangles(sNumTriangleIndices / 3);
	for(i = 0; i < sNumTriangleIndices; i++) {
        idx[i] = sMeshTriangleIndices[i];
    }
    
    btBvhTriangleMeshShape* shp = createCollisionShape();
	
	Matrix44F trans;
	
	btRigidBody * bd = PhysicsState::engine->createRigidBody(shp, trans, 0.f);
	bd->setFriction(.768f);
}
}

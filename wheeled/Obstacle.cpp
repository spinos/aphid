#include "Obstacle.h"
#include <DynamicsSolver.h>
#include "PhysicsState.h"
#include <Common.h>
namespace caterpillar {
Obstacle::Obstacle() {}
Obstacle::~Obstacle() {}
void Obstacle::create(const float & dim)
{
    btCollisionShape* rollShape = PhysicsState::engine->createCylinderShape(.372f, 32.f, .372f);
	Matrix44F tm;
	tm.rotateZ(PI * .5f);
	for(int i =0; i < 1024; i++) {
	    tm.rotateY(PI * (rand() % 399) / 399.f);
	    tm.setTranslation(((rand() % 499) / 499.f - .5f) * 2.f * dim, 0.f, ((rand() % 599) / 599.f - .5f) * 2.f * dim);
	    btTransform trans = Common::CopyFromMatrix44F(tm);
	    PhysicsState::engine->createRigidBody(rollShape, trans, 0.f);
	}
}
}

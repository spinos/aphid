#include "CollisionQuery.h"
#include <CUDABuffer.h>

#define COLLISION_MAX_OVERLAPS_PER_PRIMITIVE 24

CollisionQuery::CollisionQuery()
{ m_overlapBuffer = new CUDABuffer; }

CollisionQuery::~CollisionQuery() {}

void CollisionQuery::setNumPrimitives(unsigned n)
{ m_numPrimitives = n; }

void CollisionQuery::initOnDevice()
{ m_overlapBuffer->create(numPrimitives() * COLLISION_MAX_OVERLAPS_PER_PRIMITIVE * 4); }

const unsigned CollisionQuery::numPrimitives() const
{ return m_numPrimitives; }

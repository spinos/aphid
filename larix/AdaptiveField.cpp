#include "AdaptiveField.h"
#include <KdIntersection.h>
#include <GjkIntersection.h>

AdaptiveField::AdaptiveField(const BoundingBox & bound) :
    AdaptiveGrid(bound)
{
}

AdaptiveField::~AdaptiveField() 
{
}

void AdaptiveField::create(KdIntersection * tree, int maxLevel)
{
	BoundingBox box;
	AdaptiveGrid::create(tree, maxLevel);
	sdb::CellHash * c = cells();
	c->begin();
	while(!c->end()) {
		box = cellBox(c->key(), c->value()->level);
		gjk::IntersectTest::SetA(box);
		if(tree->intersectBox(box))
			c->value()->visited = 1;
		else
			c->value()->visited = 0;			
		c->next();
	}
	
}
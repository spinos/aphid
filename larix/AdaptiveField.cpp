#include "AdaptiveField.h"
#include <KdIntersection.h>
#include <GjkIntersection.h>
#include <ATetrahedronMesh.h>

AdaptiveField::AdaptiveField(const BoundingBox & bound) :
    AdaptiveGrid(bound)
{
}

AdaptiveField::~AdaptiveField() 
{
}

void AdaptiveField::create(KdIntersection * tree, AField * source,
					ATetrahedronMesh * mesh,
					SamplerType st,
					int maxLevel)
{
	AdaptiveGrid::create(tree, maxLevel);	
	setCellValues(tree, source, mesh, st);
}

void AdaptiveField::setCellValues(KdIntersection * tree, 
					AField * source,
					ATetrahedronMesh * mesh,
					SamplerType st)
{
	const unsigned n = numCells();
	std::vector<std::string > names;
	source->getChannelNames(names);
	std::vector<std::string >::const_iterator it = names.begin();
	for(;it!=names.end();++it) {
		source->useChannel(*it);
		if(source->currentChannel()->valueType() == TypedBuffer::TFlt)
			addFloatChannel(*it, n);
		else if(source->currentChannel()->valueType() == TypedBuffer::TVec3)
			addVec3Channel(*it, n);
	}
	
	Geometry::ClosestToPointTestResult cls;
	std::vector<unsigned> elms;
	unsigned i = 0;
	BoundingBox box;
	sdb::CellHash * c = cells();
	c->begin();
	while(!c->end()) {
		box = cellBox(c->key(), c->value()->level);
		gjk::IntersectTest::SetA(box);
		tree->countElementIntersectBox(elms, box);
		if(elms.size() > 0) {
			c->value()->visited = 1;
			cls.reset(cellCenter(c->key()), 1e8f);
			mesh->closestToPointElms(elms, &cls);
			
			setACellValues(i, &cls, source, mesh, st, names);
		}
		else
			c->value()->visited = 0;			
		c->next();
		i++;
	}
}

void AdaptiveField::setACellValues(unsigned idata, 
					Geometry::ClosestToPointTestResult * ctx,
					AField * source,
					ATetrahedronMesh * mesh,
					SamplerType st,
					const std::vector<std::string > & channelNames)
{
	std::vector<std::string >::const_iterator it = channelNames.begin();
	for(;it!=channelNames.end();++it) {
		TypedBuffer * chan = namedChannel(*it);
		if(chan->valueType() == TypedBuffer::TFlt) {
			
		}
		else if(chan->valueType() == TypedBuffer::TVec3) {
			
		}
	}
}
//:~
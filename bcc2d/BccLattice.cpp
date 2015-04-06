#include "BccLattice.h"
#include <GeoDrawer.h>
#include "bcc_common.h"
BccLattice::BccLattice(const BoundingBox & bound) :
    CartesianGrid(bound){}
BccLattice::~BccLattice() {}

void BccLattice::addOctahedron(const Vector3F & center, float h)
{
    Vector3F corner;
    int i;
    float hh = h * .5f;
	for(i=0; i < 8; i++) {
        corner = center + Vector3F(hh * OctChildOffset[i][0], 
        hh * OctChildOffset[i][1], 
        hh * OctChildOffset[i][2]);
        
        addGrid(corner);
    }
    for(i=0; i < 6; i++) {
        corner = center + Vector3F(h * HexHeighborOffset[i][0], 
        h * HexHeighborOffset[i][1], 
        h * HexHeighborOffset[i][2]);
        
        addGrid(corner);
    }
    addGrid(center);
}

void BccLattice::draw(GeoDrawer * drawer)
{
    sdb::MortonHash * latticeNode = cells();
	drawer->setColor(0.f, 0.f, 0.3f);
	float h = cellSizeAtLevel(8);
	Vector3F l;
	latticeNode->begin();
	while(!latticeNode->end()) {
	    l = gridOrigin(latticeNode->key());
	    drawer->cube(l, h);
	    latticeNode->next();
	}
}

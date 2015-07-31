#include "AdaptiveGrid.h"
#include <KdIntersection.h>
#include <GjkIntersection.h>
AdaptiveGrid::AdaptiveGrid(const BoundingBox & bound) :
    CartesianGrid(bound)
{
    
}

AdaptiveGrid::~AdaptiveGrid() 
{

}

void AdaptiveGrid::create(KdIntersection * tree)
{
// start at 8 cells per axis
    int level = 3;
    const int dim = 1<<level;
    int i, j, k;

    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;

    const Vector3F ori = origin() + Vector3F(hh, hh, hh) * .999f;
    Vector3F sample, closestP;
    BoundingBox box;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* (float)i, h* (float)j, h* (float)k);
                box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
                box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
                gjk::IntersectTest::SetA(box);
                if(tree->intersectBox(box))
                    addCell(sample, level);
            }
        }
    }
    std::cout<<" n level 3 cell "<<numCells()<<"\n";
}

#include "BccGrid.h"
#include <BezierCurve.h>
#include <GeoDrawer.h>
BccGrid::BccGrid(const BoundingBox & bound) :
    CartesianGrid(bound, 8)
{

}

BccGrid::~BccGrid() {}

void BccGrid::create(BezierCurve * curve)
{
    int level = 3;
    int dim = 1<<level;
    int i, j, k;

    const float h = cellSizeAtLevel(level);
    const Vector3F ori = origin() + Vector3F(h*.5f, h*.5f, h*.5f);
    Vector3F sample, closestP;
    float d;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* i, h* j, h* k);
                d = curve->distanceToPoint(sample, closestP);
                if(d < h)
                    addCell(i, j, k, level);
            }
        }
    }
    std::cout<<" n cell "<<numCells()<<"\n";
}

void BccGrid::draw(GeoDrawer * drawer)
{
    const unsigned n = numCells();
    unsigned i;
    Vector3F l;
    BoundingBox box;
    const float h = cellSizeAtLevel(3);
    
    for(i=0; i<n; i++) {
        l = cellOrigin(i);
        box.setMin(l.x, l.y, l.z);
        box.setMax(l.x + h, l.y + h, l.z + h);
        drawer->boundingBox(box);
    }
}

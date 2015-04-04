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
    m_tolerance = 0.33f;
    const Vector3F ori = origin() + Vector3F(h*.5f, h*.5f, h*.5f);
    Vector3F sample, closestP;
    BoundingBox box;
    float d;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* i, h* j, h* k);
                d = curve->distanceToPoint(sample, closestP);
                box.setMin(sample.x - h*.5f, sample.y - h*.5f, sample.z - h*.5f);
                box.setMax(sample.x + h*.5f, sample.y + h*.5f, sample.z + h*.5f);
                if(box.distanceTo(closestP) < m_tolerance)
                    addCell(sample, level);
            }
        }
    }
    std::cout<<" n level 3 cell "<<numCells()<<"\n";
    subdivide(curve, 4);
    // subdivide(curve, 5);
    // subdivide(curve, 6);
}

void BccGrid::subdivide(BezierCurve * curve, int level)
{
    const unsigned n = numCells();
    unsigned i;
    int u, v, w;
    Vector3F sample, subs, closestP;
    BoundingBox box;
    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;
    float d;
    int isFirst;
    for(i=0; i< n; i++) {
        sample = cellCenter(i);
        isFirst = 1;
        for(u = -1; u <= 1; u+=2) {
           for(v = -1; v <= 1; v+=2) {
               for(w = -1; w <= 1; w+=2) {
                   subs = sample + Vector3F(hh * u, hh * v, hh * w);
                   d = curve->distanceToPoint(subs, closestP);
                   box.setMin(subs.x - hh, subs.y - hh, subs.z - hh);
                   box.setMax(subs.x + hh, subs.y + hh, subs.z + hh);
                
                   if(box.distanceTo(closestP) < m_tolerance) {
                       if(isFirst) {
                           setCell(i, subs, level);
                           isFirst = 0;   
                       }
                       else 
                           addCell(subs, level);
                   }
               }
           } 
        }
    }
    std::cout<<" n level "<<level<<" cell "<<numCells()<<"\n";
}

void BccGrid::draw(GeoDrawer * drawer)
{
    const unsigned n = numCells();
    unsigned i;
    Vector3F l;
    BoundingBox box;
    const float h = cellSizeAtLevel(4) * .48f;
    
    drawer->setColor(0.f, .3f, 0.2f);
    for(i=0; i<n; i++) {
        l = cellCenter(i);
        box.setMin(l.x - h, l.y - h, l.z - h);
        box.setMax(l.x + h, l.y + h, l.z + h);
        drawer->boundingBox(box);
    }
}

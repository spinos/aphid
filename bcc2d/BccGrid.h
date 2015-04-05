#ifndef BCCGRID_H
#define BCCGRID_H

#include <CartesianGrid.h>

class BezierCurve;
class GeoDrawer;

class BccGrid : public CartesianGrid
{
public:
    BccGrid(const BoundingBox & bound);
    virtual ~BccGrid();
    
    void create(BezierCurve * curve);
    void draw(GeoDrawer * drawer);
	void drawHash();
protected:

private:
    void subdivide(BezierCurve * curve, int level);

private:
    float m_tolerance;
};
#endif        //  #ifndef BCCGRID_H


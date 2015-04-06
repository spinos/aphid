#ifndef BCCLATTICE_H
#define BCCLATTICE_H

#include <CartesianGrid.h>
class GeoDrawer;

class BccLattice : public CartesianGrid
{
public:
    BccLattice(const BoundingBox & bound);
    virtual ~BccLattice();
    
    void addOctahedron(const Vector3F & center, float h);
    
    void draw(GeoDrawer * drawer);
protected:

private:

};
#endif        //  #ifndef BCCLATTICE_H


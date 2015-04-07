#ifndef BCCLATTICE_H
#define BCCLATTICE_H

#include <CartesianGrid.h>
class GeoDrawer;

class BccLattice : public CartesianGrid
{
public:
    struct Tetrahedron {
        unsigned v[4];
    };
    
    BccLattice(const BoundingBox & bound);
    virtual ~BccLattice();
    
    void add14Node(const Vector3F & center, float h);
    void connect24Tetrahedron(const Vector3F & center, float h);
    void draw(GeoDrawer * drawer);
	
	const unsigned numGreenEdges() const;
protected:

private:
	void drawGreenEdges();

private:
    sdb::EdgeHash * m_greenEdges;
};
#endif        //  #ifndef BCCLATTICE_H


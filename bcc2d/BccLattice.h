#ifndef BCCLATTICE_H
#define BCCLATTICE_H

#include <CartesianGrid.h>
class GeoDrawer;
class BezierCurve;
class BccLattice : public CartesianGrid
{
public:
    struct Tetrahedron {
        unsigned v[4];
    };
    
    BccLattice(const BoundingBox & bound);
    virtual ~BccLattice();
    
    void add14Node(const Vector3F & center, float h);
    void prepareTetrahedron();
    void connect24Tetrahedron(const Vector3F & center, float h, BezierCurve * curve);
    void draw(GeoDrawer * drawer);
	
	const unsigned numGreenEdges() const;
	const unsigned numTetrahedrons() const;
protected:

private:
	void drawGreenEdges();
	void drawTetrahedrons();
	void encodeOctahedronVertices(const Vector3F & q, float h, int offset, unsigned * v) const;
	void add4Tetrahedrons(unsigned * vOctahedron, BezierCurve * curve);
	bool isCurveClosetToTetrahedron(const Vector3F * p, BezierCurve * curve) const;
private:
    sdb::EdgeHash * m_greenEdges;
    Tetrahedron * m_tetrahedrons;
    unsigned m_numTetrahedrons;
};
#endif        //  #ifndef BCCLATTICE_H


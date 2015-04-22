#ifndef BCCGRID_H
#define BCCGRID_H

#include <CartesianGrid.h>
struct BezierSpline;
class BccLattice;
class BezierCurve;
class GeoDrawer;

class BccGrid : public CartesianGrid
{
public:
    BccGrid(const BoundingBox & bound);
    virtual ~BccGrid();
    
    void create(BezierSpline * splines, unsigned n, int maxLevel);
	void addAnchors(unsigned * anchors, Vector3F * pos, unsigned n);
    void draw(GeoDrawer * drawer, unsigned * anchored);
	void drawHash();
	
	const unsigned numTetrahedronVertices() const;
	const unsigned numTetrahedrons() const;
	void extractTetrahedronMeshData(Vector3F * points, unsigned * indices);
protected:

private:
    void subdivide(int level);
    void createLatticeNode();
    void createLatticeTetrahedron();
	bool intersectBox(const BoundingBox & box) const;
	void drawCells(GeoDrawer * drawer);
private:
	BezierSpline * m_splines;
    BccLattice * m_lattice;
	unsigned m_numSplines;
    float m_tolerance;
};
#endif        //  #ifndef BCCGRID_H


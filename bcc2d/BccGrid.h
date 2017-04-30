#ifndef BCCGRID_H
#define BCCGRID_H

#include <CartesianGrid.h>
struct BezierSpline;
class BccLattice;
class BezierCurve;
class GeoDrawer;
class KdIntersection;

class BccGrid : public CartesianGrid
{
public:
    BccGrid(const BoundingBox & bound);
    virtual ~BccGrid();
    
    void create(KdIntersection * tree, int maxLevel);
	void addAnchors(unsigned * anchors, KdIntersection * tree);
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
	void drawCells(GeoDrawer * drawer);
private:
	KdIntersection * m_tree;
    BccLattice * m_lattice;
	float m_tolerance;
};
#endif        //  #ifndef BCCGRID_H


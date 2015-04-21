#ifndef BCCLATTICE_H
#define BCCLATTICE_H

#include <CartesianGrid.h>
class GeoDrawer;
struct BezierSpline;
class BezierCurve;
class BccLattice : public CartesianGrid
{
public:
    struct Tetrahedron {
        unsigned v[4];
    };
    
    BccLattice(const BoundingBox & bound);
    virtual ~BccLattice();
    
    void add38Node(const Vector3F & center, float h);
    void prepareTetrahedron();
    void touchIntersectedTetrahedron(const Vector3F & center, float h, 
										BezierSpline * splines, unsigned numSplines);
    void untouchGreenEdges();
    void add24Tetrahedron(const Vector3F & center, float h);
    void addNeighborTetrahedron(const Vector3F & center, float h);
    void countVisitedNodes();
    void draw(GeoDrawer * drawer);
	
	const unsigned numGreenEdges() const;
	const unsigned numTetrahedrons() const;
	const unsigned numVertices() const;
	
	void logTetrahedronMesh();
protected:

private:
	void drawGreenEdges();
	void drawTetrahedrons();
	void encodeOctahedronVertices(const Vector3F & q, float h, int offset, unsigned * v) const;
	void touch4Tetrahedrons(unsigned * vOctahedron,
							BezierSpline * splines, unsigned numSplines);
	void addTetrahedronsAllNodeVisited(unsigned * vOctahedron);
	bool isCurveClosetToTetrahedron(const Vector3F * p, BezierCurve * curve) const;
	bool intersectTetrahedron(const Vector3F * tet, BezierSpline * splines, unsigned numSplines) const;
private:
    sdb::EdgeHash * m_greenEdges;
    Tetrahedron * m_tetrahedrons;
    unsigned m_numTetrahedrons;
    unsigned m_visitedNodes;
};
#endif        //  #ifndef BCCLATTICE_H


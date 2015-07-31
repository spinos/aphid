#include <AllMath.h>
#include <BoundingBox.h>
namespace gjk {

class PointSet {
public:
    PointSet();
    virtual ~PointSet();
    
    virtual int numPoints() const;
    virtual Vector3F X(int idx) const;
    virtual Vector3F * x();
    virtual Vector3F supportPoint(const Vector3F & v, Vector3F * localP = 0) const;
protected:

private:

};

class TetrahedronSet : public PointSet {
public:
    TetrahedronSet();
    virtual ~TetrahedronSet();
    virtual int numPoints() const;
    virtual Vector3F X(int idx) const;
    virtual Vector3F * x();
private:
    Vector3F m_p[4];
};

class BoxSet : public PointSet {
public:
    BoxSet();
    virtual ~BoxSet();
    virtual int numPoints() const;
    virtual Vector3F X(int idx) const;
    virtual Vector3F * x();
private:
    Vector3F m_p[8];
};

class Simplex {
public:
    Simplex();
    
    void reset();
    void add(const Vector3F & p);
    Vector3F closestPTo(const Vector3F & referencePoint);
    bool isPInside(const Vector3F & p) const;
    void reduceToSmallest();
    
    Vector3F _p[4];
	Float4 _contributes;
    int _d;
};

class IntersectTest {
public:
    static void SetA(const BoundingBox & box);
    static bool evaluateTetrahedron(Vector3F * p, unsigned * v);
    static bool evaluate(PointSet * B);
private:
    static PointSet * A;
};
bool Intersects(PointSet * A, PointSet * B);

}

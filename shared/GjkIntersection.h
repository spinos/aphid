#pragma once
#include <AllMath.h>
#include <BoundingBox.h>

namespace aphid {

namespace gjk {

class PointSet {
public:
    PointSet();
    virtual ~PointSet();
    
    virtual int numPoints() const;
    virtual Vector3F X(int idx) const;
    virtual Vector3F * x();
    virtual Vector3F supportPoint(const Vector3F & v, Vector3F * localP = 0) const;
	const std::string str() const;
	
protected:

private:

};

class TriangleSet : public PointSet {
public:
    TriangleSet();
    virtual ~TriangleSet();
    virtual int numPoints() const;
    virtual Vector3F X(int idx) const;
    virtual Vector3F * x();
private:
    Vector3F m_p[3];
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

class Sphere {

	Vector3F m_center;
	float m_radius;
	
public:
	Sphere() {}
	
	void set(const Vector3F & c, const float & r)
	{
		m_center = c;
		m_radius = r;
	}
	
	void setCenter(const Vector3F & c)
	{ m_center = c; }
	
	void setRadius(const float & r)
	{ m_radius = r; }
    
	Vector3F supportPoint(const Vector3F & v, Vector3F * localP = 0) const
	{
		Vector3F res = m_center + v.normal() * m_radius;
		if(localP) *localP = res;
		return res;
	}
	
protected:

private:

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
    static void SetABox(const Vector3F * p);
    static void SetATetrahedron(const Vector3F * p);
    static bool evaluateTetrahedron(Vector3F * p, unsigned * v);
    static bool evaluateTriangle(Vector3F * p, unsigned * v);
    static bool evaluate(PointSet * B);
private:
    static PointSet * A;
};

template<typename T1, typename T2>
class Intersect1 {
public:
    static bool Evaluate(const T1 & A, const T2 & B)
    {
        Simplex s;
        float v2;
        Vector3F w, pa, pb, q;
    
        Vector3F v = A.X(0);
        if(v.length2() < 1e-4f) v = A.X(1);
    
        for(int i=0; i < 10; i++) {
    // SA-B(-v)
            pa = A.supportPoint(v.reversed());
            pb = B.supportPoint(v);
            w = pa - pb;
            
    // terminate when v is close enough to v(A - B).
    // http://www.bulletphysics.com/ftp/pub/test/physics/papers/jgt04raycast.pdf
            v2 = v.length2();
            if(v2 - w.dot(v) < 0.001f * v2) {
    // std::cout<<" v is close to w "<<v2 - w.dot(v)<<"\n";
                break;
            }
            
            s.add(w);
            
            if(s.isPInside(Vector3F::Zero)) {
    // std::cout<<" Minkowski difference contains the reference point\n";
                return true;
            }
            
            v = s.closestPTo(Vector3F::Zero);
            s.reduceToSmallest();
        }
        
        return false;
    }
};

}

}
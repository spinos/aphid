#include "GjkIntersection.h"
#include "tetrahedron_math.h"
namespace gjk {

PointSet::PointSet() {}
PointSet::~PointSet() {}

int PointSet::numPoints() const
{ return 0; }

Vector3F PointSet::X(int idx) const
{ return Vector3F::Zero; } 

Vector3F * PointSet::x()
{ return 0; }
    
Vector3F PointSet::supportPoint(const Vector3F & v, Vector3F * localP) const
{ 
    float maxdotv = -1e8f;
    float dotv;
	
    Vector3F res, q;
    for(int i=0; i < numPoints(); i++) {
        q = X(i);
        dotv = q.dot(v);
        if(dotv > maxdotv) {
            maxdotv = dotv;
            res = q;
            if(localP) *localP = q;
        }
    }
    
    return res;
}

TriangleSet::TriangleSet() {}
TriangleSet::~TriangleSet() {}
int TriangleSet::numPoints() const
{ return 3; }
Vector3F TriangleSet::X(int idx) const
{ return m_p[idx]; }
Vector3F * TriangleSet::x()
{ return m_p; }

TetrahedronSet::TetrahedronSet() {}

TetrahedronSet::~TetrahedronSet() {}

int TetrahedronSet::numPoints() const
{ return 4; }

Vector3F TetrahedronSet::X(int idx) const
{ return m_p[idx]; }

Vector3F * TetrahedronSet::x()
{ return m_p; }

BoxSet::BoxSet() {}

BoxSet::~BoxSet() {}

int BoxSet::numPoints() const
{ return 8; }

Vector3F BoxSet::X(int idx) const
{ return m_p[idx]; }

Vector3F * BoxSet::x()
{ return m_p; }

Simplex::Simplex() {_d = 0;}
    
void Simplex::reset() 
{ _d = 0; }

void Simplex::add(const Vector3F & p)
{
    if(_d < 1) {
        _p[0] = p;
        _d = 1;
    }
    else if(_d < 2) {
		if(p.distanceTo(_p[0]) < 1e-6f) return;
        _p[1] = p;
        _d = 2;
    }
    else if(_d < 3) {
		_p[2] = p;
		_d = 3;
		if(isTriangleDegenerated(_p)) {
			_d--;
		}
    }
    else {
        _p[3] = p;
        _d = 4;
		if(isTetrahedronDegenerated(_p)) {
		    _d--;
		}
    }
}

Vector3F Simplex::closestPTo(const Vector3F & referencePoint)
{
    Vector3F result;
    if(_d == 1) {
        result = _p[0];
        _contributes.x = 1.f;
        _contributes.y = _contributes.z = _contributes.w = 0.f;
    }
    else if(_d == 2) {
        result = closestPOnLine(_p, referencePoint);
        _contributes = getBarycentricCoordinate2(result, _p);
    }
    else if(_d == 3) {
		result = closestPOnTriangle(_p, referencePoint);
        _contributes = getBarycentricCoordinate3(result, _p);
    }
	else {
		result = closestPOnTetrahedron(_p, referencePoint);
        _contributes = getBarycentricCoordinate4(result, _p);
    }
    return result;
}

bool Simplex::isPInside(const Vector3F & referencePoint) const
{ 
    if(_d < 4) return false;
    return pointInsideTetrahedronTest(referencePoint, _p);
}

void Simplex::reduceToSmallest()
{
	if(_d < 2) return;

	float * bar = &_contributes.x;
	for(int i = 0; i < _d; i++) {
		if(bar[i] < 1e-6) {
			for(int j = i; j < _d - 1; j++) {
				_p[j] = _p[j+1];
				bar[j] = bar[j+1];
			}
			i--;
			_d--;
		}
    }
}

PointSet * IntersectTest::A = 0;

bool IntersectTest::evaluate(PointSet * B)
{
    Simplex s;
    float v2;
	Vector3F w, pa, pb, q;
    
    Vector3F v = A->X(0);
	if(v.length2() < 1e-6f) v = A->X(1);
    
    for(int i=0; i < 32; i++) {
// SA-B(-v)
	    pa = A->supportPoint(v.reversed());
		pb = B->supportPoint(v);
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

void IntersectTest::SetA(const BoundingBox & box)
{
    if(A) delete A;
    A = new BoxSet;
    Vector3F * p = A->x();
    p[0].set(box.m_data[0], box.m_data[1], box.m_data[2]);
    p[1].set(box.m_data[3], box.m_data[1], box.m_data[2]);
    p[2].set(box.m_data[0], box.m_data[4], box.m_data[2]);
    p[3].set(box.m_data[3], box.m_data[4], box.m_data[2]);
    p[4].set(box.m_data[0], box.m_data[1], box.m_data[5]);
    p[5].set(box.m_data[3], box.m_data[1], box.m_data[5]);
    p[6].set(box.m_data[0], box.m_data[4], box.m_data[5]);
    p[7].set(box.m_data[3], box.m_data[4], box.m_data[5]);
}

void IntersectTest::SetATetrahedron(const Vector3F * p)
{
    if(A) delete A;
    A = new TetrahedronSet;
    A->x()[0] = p[0];
    A->x()[1] = p[1];
    A->x()[2] = p[2];
    A->x()[3] = p[3];
}

void IntersectTest::SetABox(const Vector3F * p)
{
    if(A) delete A;
    A = new BoxSet;
    Vector3F * q = A->x();
    q[0] = p[0];
    q[1] = p[1];
    q[2] = p[2];
    q[3] = p[3];
    q[4] = p[4];
    q[5] = p[5];
    q[6] = p[6];
    q[7] = p[7];
}

bool IntersectTest::evaluateTetrahedron(Vector3F * p, unsigned * v)
{
    TetrahedronSet B;
    Vector3F * pt = B.x();
    pt[0] = p[v[0]];
    pt[1] = p[v[1]];
    pt[2] = p[v[2]];
    pt[3] = p[v[3]];
    return evaluate(&B);
}

bool IntersectTest::evaluateTriangle(Vector3F * p, unsigned * v)
{
    TriangleSet B;
    Vector3F * pt = B.x();
    pt[0] = p[v[0]];
    pt[1] = p[v[1]];
    pt[2] = p[v[2]];
    return evaluate(&B);
}

}

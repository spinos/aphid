#include <Gjk.h>

float determinantTetrahedron(Matrix44F & mat, const Vector3F & v1, const Vector3F & v2, const Vector3F & v3, const Vector3F & v4)
{
    * mat.m(0, 0) = v1.x;
    * mat.m(0, 1) = v1.y;
    * mat.m(0, 2) = v1.z;
    * mat.m(0, 3) = 1.f;
    
    * mat.m(1, 0) = v2.x;
    * mat.m(1, 1) = v2.y;
    * mat.m(1, 2) = v2.z;
    * mat.m(1, 3) = 1.f;
    
    * mat.m(2, 0) = v3.x;
    * mat.m(2, 1) = v3.y;
    * mat.m(2, 2) = v3.z;
    * mat.m(2, 3) = 1.f;
    
    * mat.m(3, 0) = v4.x;
    * mat.m(3, 1) = v4.y;
    * mat.m(3, 2) = v4.z;
    * mat.m(3, 3) = 1.f;
    
    return mat.determinant();
}

char isTetrahedronDegenerate(const Vector3F * v)
{
    Matrix44F mat;
    float D0 = determinantTetrahedron(mat, v[0], v[1], v[2], v[3]);
	if(D0 < 0.f) D0 = -D0;
	// std::cout<<" D0 "<<D0<<"\n";
    return (D0 < 1e-2);
}

char isTriangleDegenerate(const Vector3F * v)
{
	Vector3F n = Vector3F(v[1], v[0]).cross( Vector3F(v[2], v[0]) );
	
    float D0 = n.length2();
	// std::cout<<" D0 "<<D0<<"\n";
    return (D0 < 1e-6);
}

BarycentricCoordinate getBarycentricCoordinate2(const Vector3F & p, const Vector3F * v)
{
	BarycentricCoordinate coord;
	Vector3F dv = v[1] - v[0];
	float D0 = dv.length();
	if(D0 == 0.f) {
        std::cout<<" line is degenerate ("<<v[0].str()<<","<<v[1].str()<<")\n";
        coord.x = coord.y = coord.z = coord.w = -1.f;
        return coord;
    }  
	Vector3F dp = p - v[1];
	coord.x = dp.length() / D0;
	coord.y = 1.f - coord.x;
	
	return coord;
}

BarycentricCoordinate getBarycentricCoordinate3(const Vector3F & p, const Vector3F * v)
{
    Matrix33F mat;
    
    BarycentricCoordinate coord;
    
    Vector3F n = Vector3F(v[1] - v[0]).cross( Vector3F(v[2] - v[0]) );
	
    float D0 = n.length2();
    if(D0 == 0.f) {
        std::cout<<" tiangle is degenerate ("<<v[0].str()<<","<<v[1].str()<<","<<v[2].str()<<")\n";
        coord.x = coord.y = coord.z = coord.w = -1.f;
        return coord;
    }  
    
	Vector3F na = Vector3F(v[2] - v[1]).cross( Vector3F(p - v[1]) );
    float D1 = n.dot(na);
	Vector3F nb = Vector3F(v[0] - v[2]).cross( Vector3F(p - v[2]) );  
    float D2 = n.dot(nb);
	Vector3F nc = Vector3F(v[1] - v[0]).cross( Vector3F(p - v[0]) );
    float D3 = n.dot(nc);
    
    coord.x = D1/D0;
    coord.y = D2/D0;
    coord.z = D3/D0;
    coord.w = 1.f;
    
    return coord;
}

BarycentricCoordinate getBarycentricCoordinate4(const Vector3F & p, const Vector3F * v)
{
    Matrix44F mat;
    
    BarycentricCoordinate coord;
    
    float D0 = determinantTetrahedron(mat, v[0], v[1], v[2], v[3]);
    if(D0 == 0.f) {
        std::cout<<" tetrahedron is degenerate ("<<v[0].str()<<","<<v[1].str()<<","<<v[2].str()<<","<<v[3].str()<<")\n";
        coord.x = coord.y = coord.z = coord.w = -1.f;
        return coord;
    }  
    
    float D1 = determinantTetrahedron(mat, p, v[1], v[2], v[3]);
    float D2 = determinantTetrahedron(mat, v[0], p, v[2], v[3]);
    float D3 = determinantTetrahedron(mat, v[0], v[1], p, v[3]);
    float D4 = determinantTetrahedron(mat, v[0], v[1], v[2], p);
    
    coord.x = D1/D0;
    coord.y = D2/D0;
    coord.z = D3/D0;
    coord.w = D4/D0;
    
    return coord;
}

void closestOnLine(const Vector3F * p, ClosestTestContext * io)
{
    Vector3F vr = io->referencePoint - p[0];
    Vector3F v1 = p[1] - p[0];
	const float dr = vr.length();
	if(dr < TINY_VALUE) {
        io->closestPoint = p[0];
		io->distance = 0.f;
        return;
    }
	
	const float d1 = v1.length();
	vr.normalize();
	v1.normalize();
	float vrdv1 = vr.dot(v1) * dr;
	if(vrdv1 < 0.f) vrdv1 = 0.f;
	if(vrdv1 > d1) vrdv1 = d1;
	
	v1 = p[0] + v1 * vrdv1;
	const float dc = v1.distanceTo(io->referencePoint);
	
	if(dc > io->distance) return;
	
	io->hasResult = 1;
	io->closestPoint = v1;
	io->distance = dc;
}

void closestPointToOriginInsideTriangle(const Vector3F * p, ClosestTestContext * io)
{// std::cout<<" p in tri test ";
	io->hasResult = 0;
    Vector3F ab = p[1] - p[0];
    Vector3F ac = p[2] - p[0];
    Vector3F nor = ab.cross(ac);
    nor.normalize();
    
    float t = p[0].dot(nor);
    Vector3F onplane = nor * t;
    
    Vector3F e01 = p[1] - p[0];
	Vector3F x0 = onplane - p[0];
	if(e01.cross(x0).dot(nor) < 0.f) return;
	
	Vector3F e12 = p[2] - p[1];
	Vector3F x1 = onplane - p[1];
	if(e12.cross(x1).dot(nor) < 0.f) return;
	
	Vector3F e20 = p[0] - p[2];
	Vector3F x2 = onplane - p[2];
	if(e20.cross(x2).dot(nor) < 0.f) return;
	
	const float dc = onplane.length();
	if(dc > io->distance) return;
	
	io->hasResult = 1;
	io->closestPoint = onplane + io->referencePoint;
	io->distance = dc;
}

void printPoints(const Vector3F * p, int n)
{
	for(int i = 0; i < n; i++)
		std::cout<<p[i].str();
	std::cout<<"\n";
}

void closestOnTriangle(const Vector3F * p, ClosestTestContext * io)
{	
// std::cout<<" closest on tri test ";
// printPoints(p, 3);
	Vector3F pr[3];
	pr[0] = p[0] - io->referencePoint;
	pr[1] = p[1] - io->referencePoint;
	pr[2] = p[2] - io->referencePoint;
	
    closestPointToOriginInsideTriangle(pr, io);
	
	if(io->hasResult) return;
		
// std::cout<<" p on tri edge test ";
	pr[0] = p[0];
	pr[1] = p[1];
	closestOnLine(pr, io);
	
	pr[0] = p[1];
	pr[1] = p[2];
	closestOnLine(pr, io);
	
	pr[0] = p[2];
	pr[1] = p[0];
	closestOnLine(pr, io);
}

void checkTet(const Vector3F * p, ClosestTestContext * io)
{
	float sum = io->contributes.x + io->contributes.y + io->contributes.z + io->contributes.w;
	if(sum > 1.01f) {
		std::cout<<" tet contrib "<<io->contributes.x<<" "<<io->contributes.y<<" "<<io->contributes.z<<" "<<io->contributes.w;
		std::cout<<" "<<sum<<"\n";
		std::cout<<" ref point "<<io->referencePoint.str()<<"\n";
		std::cout<<" closet point "<<io->closestPoint.str()<<"\n";
		std::cout<<" tex point "<<p[0].str()<<" "<<p[1].str()<<" "<<p[2].str()<<" "<<p[3].str()<<"\n";
		
		Matrix44F mat;
		float D0 = determinantTetrahedron(mat, p[0], p[1], p[2], p[3]);
		std::cout<<" det "<<D0<<"\n";
	}
}

void closestOnTetrahedron(const Vector3F * p, ClosestTestContext * io)
{
// std::cout<<" closest on tet test ";
// printPoints(p, 4);
	closestOnTriangle(p, io);
	
	Vector3F pr[3];
	pr[0] = p[0];
	pr[1] = p[1];
	pr[2] = p[3];
	closestOnTriangle(pr, io);
	
	pr[0] = p[0];
	pr[1] = p[2];
	pr[2] = p[3];
	closestOnTriangle(pr, io);
	
	pr[0] = p[1];
	pr[1] = p[2];
	pr[2] = p[3];
	closestOnTriangle(pr, io);
}

void resetSimplex(Simplex & s)
{
    s.d = 0;
}

void addToSimplex(Simplex & s, const Vector3F & p)
{// std::cout<<"\n add\n";
    if(s.d < 1) {
        s.p[0] = p;
        s.d = 1;
    }
    else if(s.d < 2) {
		if(p.distanceTo(s.p[0]) < TINY_VALUE) return;
        s.p[1] = p;
        s.d = 2;
    }
    else if(s.d < 3) {
		s.p[2] = p;
		s.d = 3;
		if(isTriangleDegenerate(s.p)) {
			// std::cout<<" degenerate triangle";
			// printTri(s.p);
			s.d--;
		}
		// else { std::cout<<" new tri ";
		//	printTri(s.p);
		//}
    }
    else {
        s.p[3] = p;
        s.d = 4;
		if(isTetrahedronDegenerate(s.p)) {
			// std::cout<<" degenerate tetrahedron";
		    // printTet(s.p);
		    s.d--;
		}
		// else  { std::cout<<" new tet ";
			// printTet(s.p);
		// }
    }
}

void addToSimplex(Simplex & s, const Vector3F & p, const Vector3F & pb)
{// std::cout<<"\n add pb\n";
    if(s.d < 1) {
        s.p[0] = p;
		s.pB[0] = pb;
        s.d = 1;
    }
    else if(s.d < 2) {
		if(p.distanceTo(s.p[0]) < TINY_VALUE) return;
        s.p[1] = p;
		s.pB[1] = pb;
        s.d = 2;
    }
    else if(s.d < 3) {
		s.p[2] = p;
		s.pB[2] = pb;
		s.d = 3;
		if(isTriangleDegenerate(s.p)) {
			// std::cout<<" degenerate triangle";
			// printTri(s.p);
			s.d--;
		}
    }
    else {
        s.p[3] = p;
		s.pB[3] = pb;
        s.d = 4;
		if(isTetrahedronDegenerate(s.p)) {
			// std::cout<<" degenerate tetrahedron";
		    // printPoints(s.p, 4);
		    s.d--;
		}
    }
}

void smallestSimplex(ClosestTestContext * io)
{
	Simplex & s = io->W;
    if(s.d < 2) return;
	// int od = s.d;
	float * bar = &io->contributes.x;
	for(int i = 0; i < s.d; i++) {
		if(bar[i] < TINY_VALUE) {
			// std::cout<<" zero "<<bar[i]<<" remove vertex "<<i<<"\n";
			for(int j = i; j < s.d - 1; j++) {
				s.p[j] = s.p[j+1];
				s.pB[j] = s.pB[j+1];
				bar[j] = bar[j+1];
			}
			i--;
			s.d--;
		}
    }
	// if(s.d < od) {
	// 	std::cout<<"  reduce from "<<od<<" to "<<s.d<<"\n";
	// 	for(int i = 0; i < s.d; i++) {
	// 		std::cout<<s.p[i].str();
	// 	}
	// 	std::cout<<"\n";
	// }
}

char pointInsideTetrahedronTest(const Vector3F & p, const Vector3F * v)
{
    if(isTetrahedronDegenerate(v))		
        return 0;
        
    BarycentricCoordinate coord = getBarycentricCoordinate4(p, v);
    // std::cout<<"sum "<<coord.x + coord.y + coord.z + coord.w<<"\n";
    
    //Vector3F proof = v[0] * coord.x + v[1] * coord.y + v[2] * coord.z + v[3] * coord.w;
    //std::cout<<"proof "<<proof.str()<<"\n";
    
    if(coord.x < 0.f || coord.y < 0.f || coord.z < 0.f || coord.w < 0.f)
        return 0;
    
    if(coord.x > 1.f || coord.y > 1.f || coord.z > 1.f || coord.w > 1.f)
        return 0;
    
    return 1;
}

char isPointInsideSimplex(const Simplex & s, const Vector3F & p)
{
    if(s.d < 4) return 0;
    return pointInsideTetrahedronTest(p, s.p);
}

void closestOnSimplex(ClosestTestContext * io)
{
	Simplex & s = io->W;
    if(s.d == 1) {
        io->closestPoint = s.p[0];
        io->distance = io->closestPoint.distanceTo(io->referencePoint);
    }
    else if(s.d == 2)
        closestOnLine(s.p, io);
    else if(s.d == 3)
		closestOnTriangle(s.p, io);
	else
		closestOnTetrahedron(s.p, io);
		
	if(!io->needContributes) return;
	
	if(s.d == 1) 
		io->contributes.x = 1.f;
	else if(s.d == 2) 
		io->contributes = getBarycentricCoordinate2(io->closestPoint, s.p);
	else if(s.d == 3)
		io->contributes = getBarycentricCoordinate3(io->closestPoint, s.p);
	else 
		io->contributes = getBarycentricCoordinate4(io->closestPoint, s.p);
}

Vector3F supportMapping(const PointSet & A, const PointSet & B, const Vector3F & v)
{
	return (A.supportPoint(v) - B.supportPoint(v.reversed()));
}

void interpolatePointB(ClosestTestContext * io)
{
	Simplex & s = io->W;
	// std::cout<<" contact b "<<s.d<<" ";
	Vector3F sum = Vector3F::Zero;
	const float * wei = &io->contributes.x;
	float one = 0.f;
	for(int i =0; i < s.d; i++) {
		// std::cout<<" "<<wei[i];
		if(wei[i] > TINY_VALUE) {
			sum += s.pB[i] * wei[i];
			one += wei[i];
		}
	}
	io->contactPointB = sum;
	// std::cout<<" "<<one<<"\n";
}

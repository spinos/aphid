/*
 *  BaseMesh.cpp
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <BarycentricCoordinate.h>
#include "BaseMesh.h"

BaseMesh::BaseMesh()
{
	_vertices = 0;
	_indices = 0;
	m_normals = 0;
	m_quadIndices = 0;
	m_polygonCounts = 0;
	m_polygonIndices = 0;
	m_numPolygons = 0;
	m_u = 0;
	m_v = 0;
	m_uvIds = 0;
	m_numUVs = m_numUVIds = 0;
	setEntityType(TypedEntity::TTriangleMesh);
}

BaseMesh::~BaseMesh()
{
	cleanup();
}

bool BaseMesh::isEmpty() const
{
	return _numVertices < 3;
}

void BaseMesh::cleanup()
{
	if(_vertices) {delete[] _vertices; _vertices = 0;}
	if(m_normals) {delete[] m_normals; m_normals = 0;}
	if(_indices) {delete[] _indices; _indices = 0;}
	if(m_quadIndices) {delete[] m_quadIndices; m_quadIndices = 0;}
	if(m_polygonCounts) {delete[] m_polygonCounts; m_polygonCounts = 0;}
	if(m_polygonIndices) {delete[] m_polygonIndices; m_polygonIndices = 0;}
	if(m_u) {delete[] m_u; m_u = 0;}
	if(m_v) {delete[] m_v; m_v = 0;}
	if(m_uvIds) {delete[] m_uvIds; m_uvIds = 0;}
	_numVertices = 0;
	m_numTriangles = 0;
	m_numTriangleFaceVertices = 0;
	m_numPolygons = 0;
	m_numUVs = m_numUVIds = 0;
}

void BaseMesh::createVertices(unsigned num)
{
	_vertices = new Vector3F[num];
	m_normals = new Vector3F[num];
	_numVertices = num;
}

void BaseMesh::createIndices(unsigned num)
{
	_indices = new unsigned[num];
	m_numTriangleFaceVertices = num;
}

void BaseMesh::createQuadIndices(unsigned num)
{
	m_quadIndices = new unsigned[num];
	m_numQuadVertices = num;
}

void BaseMesh::createPolygonCounts(unsigned num)
{
    m_numPolygons = num;
    m_polygonCounts = new unsigned[num];
}

void BaseMesh::createPolygonIndices(unsigned num)
{
    m_numPolygonVertices = num;
    m_polygonIndices = new unsigned[num];
}

void BaseMesh::createPolygonUV(unsigned numUVs, unsigned numUVIds)
{
	m_u = new float[numUVs];
	m_v = new float[numUVs];
	m_uvIds = new unsigned[numUVIds];
	m_numUVs = numUVs;
	m_numUVIds = numUVIds;
}

unsigned BaseMesh::processTriangleFromPolygon()
{
	m_numTriangles = 0;
	
	unsigned i, j;
    for(i = 0; i < m_numPolygons; i++)
        m_numTriangles += m_polygonCounts[i] - 2;
		
	createIndices(m_numTriangles * 3);
    
    unsigned curTri = 0;
    unsigned curFace = 0;
    for(i = 0; i < m_numPolygons; i++) {
        for(j = 1; j < m_polygonCounts[i] - 1; j++) {
            _indices[curTri] = m_polygonIndices[curFace];
            _indices[curTri + 1] = m_polygonIndices[curFace + j];
            _indices[curTri + 2] = m_polygonIndices[curFace + j + 1];
            curTri += 3;
        }
        curFace += m_polygonCounts[i];
    }
	
	return m_numTriangles;
}

unsigned BaseMesh::processQuadFromPolygon()
{
	unsigned i, j;
	m_numQuads = 0;
    for(i = 0; i < m_numPolygons; i++) {
		if(m_polygonCounts[i] < 5)
			m_numQuads++;
	}
		
	if(m_numQuads < 1) return 0;
	
	createQuadIndices(m_numQuads * 4);
	
	unsigned ie = 0;
	unsigned curFace = 0;
	for(i = 0; i < m_numPolygons; i++) {
		if(m_polygonCounts[i] == 4) {
			for(j = 0; j < 4; j++) {
				m_quadIndices[ie] = m_polygonIndices[curFace + j];
				ie++;
			}
		}
		else if(m_polygonCounts[i] == 3) {
			for(j = 0; j < 3; j++) {
				m_quadIndices[ie] = m_polygonIndices[curFace + j];
				ie++;
			}
			m_quadIndices[ie] = m_quadIndices[ie - 3];
			ie++;
		}
		
		curFace += m_polygonCounts[i];
	}
	
	return m_numQuads;
}

const BoundingBox BaseMesh::calculateBBox() const
{
	BoundingBox box;
	const unsigned nv = getNumVertices();
	for(unsigned i = 0; i < nv; i++) {
		box.updateMin(_vertices[i]);
		box.updateMax(_vertices[i]);
	}
	return box;
}

const BoundingBox BaseMesh::calculateBBox(const unsigned &idx) const
{
	BoundingBox box;
	unsigned *trii = &_indices[idx * 3];
	Vector3F *p0 = &_vertices[*trii];
	trii++;
	Vector3F *p1 = &_vertices[*trii];
	trii++;
	Vector3F *p2 = &_vertices[*trii];
		
	box.updateMin(*p0);
	box.updateMax(*p0);
	box.updateMin(*p1);
	box.updateMax(*p1);
	box.updateMin(*p2);
	box.updateMax(*p2);

	return box;
}

const int BaseMesh::faceOnSideOf(const unsigned &idx, const int &axis, const float &pos) const
{
	unsigned *trii = &_indices[idx * 3];
	Vector3F *p0 = &_vertices[*trii];
	trii++;
	Vector3F *p1 = &_vertices[*trii];
	trii++;
	Vector3F *p2 = &_vertices[*trii];
	if(axis == 0) {
		if(p0->x < pos && p1->x < pos && p2->x < pos)
			return 0;
		if(p0->x >= pos && p1->x >= pos && p2->x >= pos)
			return 2;
		return 1;
	}
	else if(axis == 1) {
		if(p0->y < pos && p1->y < pos && p2->y < pos)
			return 0;
		if(p0->y >= pos && p1->y >= pos && p2->y >= pos)
			return 2;
		return 1;
	}
	if(p0->z < pos && p1->z < pos && p2->z < pos)
		return 0;
	if(p0->z >= pos && p1->z >= pos && p2->z >= pos)
		return 2;
	return 1;
}

Vector3F * BaseMesh::vertices()
{
	return _vertices;
}

Vector3F * BaseMesh::normals()
{
	return m_normals;
}

unsigned * BaseMesh::indices()
{
	return _indices;
}

unsigned * BaseMesh::quadIndices()
{
    return m_quadIndices;
}

unsigned * BaseMesh::polygonCounts()
{
	return m_polygonCounts;
}

unsigned * BaseMesh::polygonIndices()
{
	return m_polygonIndices;
}

float * BaseMesh::us()
{
	return m_u;
}

float * BaseMesh::vs()
{
	return m_v;
}

unsigned * BaseMesh::uvIds()
{
	return m_uvIds;
}

void BaseMesh::setVertex(unsigned idx, float x, float y, float z)
{
	Vector3F *v = _vertices;
	v += idx;
	v->x = x;
	v->y = y;
	v->z = z;
}

void BaseMesh::setTriangle(unsigned idx, unsigned a, unsigned b, unsigned c)
{
	unsigned *i = _indices;
	i += idx * 3;
	*i = a;
	i++;
	*i = b;
	i++;
	*i = c;
}

void BaseMesh::move(float x, float y, float z)
{
	const unsigned nv = getNumVertices();
	for(unsigned i = 0; i < nv; i++) {
		_vertices[i].x += x;
		_vertices[i].y += y;
		_vertices[i].y += z;
	}
}

unsigned BaseMesh::getNumFaces() const
{
	return getNumTriangles();
}

unsigned BaseMesh::getNumTriangles() const
{
	return m_numTriangles;
}

unsigned BaseMesh::getNumQuads() const
{
	return m_numQuads;
}

unsigned BaseMesh::getNumPolygons() const
{
	return m_numPolygons;
}

unsigned BaseMesh::getNumVertices() const
{
	return _numVertices;
}

unsigned BaseMesh::getNumPolygonFaceVertices() const
{
	return m_numPolygonVertices;
}

unsigned BaseMesh::getNumTriangleFaceVertices() const
{
	return m_numTriangleFaceVertices;
}

unsigned BaseMesh::getNumUVs() const
{
	return m_numUVs;
}

unsigned BaseMesh::getNumUVIds() const
{
	return m_numUVIds;
}

Vector3F * BaseMesh::getVertices() const
{
	return _vertices;
}

Vector3F * BaseMesh::getNormals() const
{
	return m_normals;
}

unsigned * BaseMesh::getIndices() const
{
	return _indices;
}

unsigned * BaseMesh::getQuadIndices() const
{
	return m_quadIndices;
}

unsigned * BaseMesh::getPolygonCounts() const
{
	return m_polygonCounts;
}

unsigned * BaseMesh::getPolygonIndices() const
{
	return m_polygonIndices;
}

float * BaseMesh::getUs() const
{
	return m_u;
}

float * BaseMesh::getVs() const
{
	return m_v;
}

unsigned * BaseMesh::getUvIds() const
{
	return m_uvIds;
}

Matrix33F BaseMesh::getTangentFrame(const unsigned& idx) const
{
	return Matrix33F();
}

char BaseMesh::intersect(unsigned idx, IntersectionContext * ctx) const
{
    Vector3F threeCorners[3];
	threeCorners[0] = _vertices[_indices[idx * 3]];
	threeCorners[1] = _vertices[_indices[idx * 3 + 1]];
	threeCorners[2] = _vertices[_indices[idx * 3 + 2]];
	
	if(!triangleIntersect(threeCorners, ctx)) return 0;

	postIntersection(idx, ctx);
	
	return 1;
}

void BaseMesh::postIntersection(unsigned idx, IntersectionContext * ctx) const
{
    if(ctx->getComponentFilterType() == PrimitiveFilter::TFace)
	    ctx->m_componentIdx = idx;
	else 
	    ctx->m_componentIdx = closestVertex(idx, ctx->m_hitP);
}

char BaseMesh::naiveIntersect(IntersectionContext * ctx) const
{   
	unsigned nf = BaseMesh::getNumFaces();
	for(unsigned i = 0; i < nf; i++) {
		if(BaseMesh::intersect(i, ctx)) return 1;
	}
	return 0;
}

void BaseMesh::getTriangle(unsigned idx, unsigned *vertexId) const
{
	unsigned *i = _indices;
	i += idx * 3;
	vertexId[0] = *i;
	i++;
	vertexId[1] = *i;
	i++;
	vertexId[2] = *i;
}

char BaseMesh::closestPoint(unsigned idx, const Vector3F & origin, IntersectionContext * ctx) const
{
	Vector3F a = _vertices[_indices[idx * 3]];
	Vector3F b = _vertices[_indices[idx * 3 + 1]];
	Vector3F c = _vertices[_indices[idx * 3 + 2]];
	
	BarycentricCoordinate bar;
	bar.create(a, b, c);
	float d = bar.project(origin);
	if(d >= ctx->m_minHitDistance) return 0;
	
	if(ctx->m_enableNormalRef) {
		if(bar.getNormal().dot(ctx->m_refN) < 0.1f) return 0;
	}
	
	bar.compute();
	if(!bar.insideTriangle()) {
		bar.computeClosest();
	}
	
	char hit = 0;
	Vector3F clampledP = bar.getClosest();
	d = (clampledP - origin).length();

	if(d < ctx->m_minHitDistance) {
		ctx->m_minHitDistance = d;
		ctx->m_componentIdx = idx;
		ctx->m_coord[0] = bar.getV(0);
		ctx->m_coord[1] = bar.getV(1);
		ctx->m_coord[2] = bar.getV(2);
		ctx->m_closestP = clampledP;
		
		ctx->m_hitP = clampledP;
		hit = 1;
	}
	return hit;
}

char BaseMesh::insideTriangle(const Vector3F & p, const Vector3F & a, const Vector3F & b, const Vector3F & c, const Vector3F & n) const
{
	Vector3F e01 = b - a;
	Vector3F x0 = p - a;
	if(e01.cross(x0).dot(n) < 0.f) return 0;
	
	Vector3F e12 = c - b;
	Vector3F x1 = p - b;
	if(e12.cross(x1).dot(n) < 0.f) return 0;
	
	Vector3F e20 = a - c;
	Vector3F x2 = p - c;
	if(e20.cross(x2).dot(n) < 0.f) return 0;
	
	return 1;
}

unsigned BaseMesh::closestVertex(unsigned idx, const Vector3F & px) const
{
	unsigned *trii = &_indices[idx * 3];
	float mag, minDist = 10e8;
	unsigned vert = 0;
	for(int i = 0; i < 3; i++) {
		Vector3F v = _vertices[*trii] - px;
		
		mag = v.length();
		
		if(mag < minDist) {
			minDist = mag;
			vert = *trii;
		}
		trii++;
	}
	return vert;
}

char BaseMesh::triangleIntersect(const Vector3F * threeCorners, IntersectionContext * ctx) const
{
    Vector3F a = threeCorners[0];
	Vector3F b = threeCorners[1];
	Vector3F c = threeCorners[2];
	Vector3F ab = b - a;
	Vector3F ac = c - a;
	Vector3F nor = ab.cross(ac);
	nor.normalize();
	
	Ray &ray = ctx->m_ray;
	float ddotn = ray.m_dir.dot(nor);
		
	if(ddotn > 0.f) return 0;
	
	float t = (a.dot(nor) - ray.m_origin.dot(nor)) / ddotn;
	
	if(t < 0.f || t > ray.m_tmax) return 0;
	
	//printf("face %i %f %f", idx, t, ctx->m_minHitDistance);
	
	if(t > ctx->m_minHitDistance) return 0;
	
	Vector3F onplane = ray.m_origin + ray.m_dir * t;
	Vector3F e01 = b - a;
	Vector3F x0 = onplane - a;
	if(e01.cross(x0).dot(nor) < 0.f) return 0;
	
	//printf("pass a\n");

	Vector3F e12 = c - b;
	Vector3F x1 = onplane - b;
	if(e12.cross(x1).dot(nor) < 0.f) return 0;
	
	//printf("pass b\n");
	
	Vector3F e20 = a - c;
	Vector3F x2 = onplane - c;
	if(e20.cross(x2).dot(nor) < 0.f) return 0;
	
	//printf("pass c\n");
	
	ctx->m_hitP = onplane;
	ctx->m_hitN = nor;
	ctx->m_minHitDistance = t;
	ctx->m_geometry = (Geometry*)this;
	ctx->m_success = 1;
	return 1;
}

void BaseMesh::putIntoObjectSpace()
{
	BoundingBox b = BaseMesh::calculateBBox();
	setBBox(b);
	
	const Vector3F c = b.center();
	
	const unsigned nv = getNumVertices();
	for(unsigned i = 0; i < nv; i++)
		_vertices[i] -= c;
}

void BaseMesh::verbose() const
{
	std::cout<<"mesh status:\n";
	std::cout<<" num vertices "<<getNumVertices()<<"\n";
	std::cout<<" num polygons "<<getNumPolygons()<<"\n";
	std::cout<<" num polygon face vertices "<<getNumPolygonFaceVertices()<<"\n";
	std::cout<<" num triangles "<<getNumTriangles()<<"\n";
	std::cout<<" num quads "<<getNumQuads()<<"\n";
	std::cout<<" num uvs "<<getNumUVs()<<"\n";
	std::cout<<" num uvs indices "<<getNumUVIds()<<"\n";
}
//:~

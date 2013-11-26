/*
 *  BaseMesh.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <Geometry.h>
#include <IntersectionContext.h>
#include <Ray.h>

class BaseMesh : public Geometry {
public:
	BaseMesh();
	virtual ~BaseMesh();
	
	bool isEmpty() const;
	void cleanup();
	
	void createVertices(unsigned num);
	void createIndices(unsigned num);
	void createQuadIndices(unsigned num);
	void createPolygonCounts(unsigned num);
	void createPolygonIndices(unsigned num);
	void createPolygonUV(unsigned numUVs, unsigned numUVIds);
	
	unsigned processTriangleFromPolygon();
	unsigned processQuadFromPolygon();

	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(const unsigned &idx) const;
	const int faceOnSideOf(const unsigned &idx, const int &axis, const float &pos) const;
	
	Vector3F * vertices();
	Vector3F * normals();
	unsigned * indices();
	unsigned * quadIndices();
	unsigned * polygonCounts();
	unsigned * polygonIndices();
	float * us();
	float * vs();
	unsigned * uvIds();
	
	void setVertex(unsigned idx, float x, float y, float z);
	void setTriangle(unsigned idx, unsigned a, unsigned b, unsigned c);
	void move(float x, float y, float z);
	
	void getTriangle(unsigned idx, unsigned *vertexId) const;
	
	virtual unsigned getNumFaces() const;
	unsigned getNumTriangles() const;
	unsigned getNumQuads() const;
	unsigned getNumPolygons() const;
	unsigned getNumVertices() const;
	unsigned getNumPolygonFaceVertices() const;
	unsigned getNumTriangleFaceVertices() const;
	unsigned getNumUVs() const;
	unsigned getNumUVIds() const;
	
	Vector3F * getVertices() const;
	Vector3F * getNormals() const;
	unsigned * getIndices() const;
	unsigned * getQuadIndices() const;
	unsigned * getPolygonCounts() const;
	unsigned * getPolygonIndices() const;
	float * getUs() const;
	float * getVs() const;
	unsigned * getUvIds() const;
	virtual Matrix33F getTangentFrame(const unsigned& idx) const;
	
	virtual char intersect(unsigned idx, IntersectionContext * ctx) const;
	void postIntersection(unsigned idx, IntersectionContext * ctx) const;
	virtual char closestPoint(unsigned idx, const Vector3F & origin, IntersectionContext * ctx) const;
	char insideTriangle(const Vector3F & p, const Vector3F & a, const Vector3F & b, const Vector3F & c, const Vector3F & n) const;
	virtual unsigned closestVertex(unsigned idx, const Vector3F & px) const;
	char triangleIntersect(const Vector3F * threeCorners, IntersectionContext * ctx) const;
	char selectComponent(IntersectionContext * ctx) const;
	
	void putIntoObjectSpace();
	
	void createPerVertexVector();
	Vector3F *perVertexVector() const;
	
	void verbose() const;
	
	Vector3F * _vertices;
	Vector3F * m_normals;
	unsigned * _indices;
	unsigned * m_quadIndices;
	unsigned _numVertices;
	unsigned m_numTriangles;
	unsigned m_numTriangleFaceVertices;
	unsigned m_numQuadVertices;
	unsigned m_numPolygons;
	unsigned m_numQuads;
	unsigned * m_polygonCounts;
	unsigned m_numPolygonVertices;
	unsigned * m_polygonIndices;
	unsigned m_numUVs, m_numUVIds;
	float * m_u;
	float * m_v;
	unsigned * m_uvIds;
private:
	Vector3F * m_pvv;
};
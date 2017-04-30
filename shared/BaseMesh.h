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
#include <geom/Geometry.h>
#include <Boundary.h>
#include <IntersectionContext.h>
#include <math/Ray.h>
#include <VertexDataGroup.h>

namespace aphid {

class BaseMesh : public Geometry, public Boundary {
public:
	BaseMesh();
	virtual ~BaseMesh();
	
	bool isEmpty() const;
	void cleanup();
	
	void createVertices(unsigned num);
	void createIndices(unsigned num);
	void createPolygonCounts(unsigned num);
	void createPolygonIndices(unsigned num);
	void createPolygonUV(unsigned numUVs, unsigned numUVIds);
	
	unsigned processTriangleFromPolygon();
	
// override geometry
	virtual const unsigned numComponents() const;
	virtual const BoundingBox calculateBBox() const;
	virtual const BoundingBox calculateBBox(unsigned icompenent) const;
	
	const int faceOnSideOf(const unsigned &idx, const int &axis, const float &pos) const;
	
	Vector3F * vertices();
	Vector3F * normals();
	unsigned * indices();
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
	const unsigned & getNumPolygons() const;
	const unsigned & getNumVertices() const;
	const unsigned & getNumPolygonFaceVertices() const;
	unsigned getNumTriangleFaceVertices() const;
	unsigned getNumUVs() const;
	unsigned getNumUVIds() const;
	
	Vector3F * getVertices() const;
	Vector3F * getNormals() const;
	unsigned * getIndices() const;
	unsigned * getPolygonCounts() const;
	unsigned * getPolygonIndices() const;
	float * getUs() const;
	float * getVs() const;
	unsigned * getUvIds() const;
	virtual Matrix33F getTangentFrame(const unsigned& idx) const;
	virtual Vector3F getFaceNormal(const unsigned & idx) const;
	
	virtual char intersect(unsigned idx, IntersectionContext * ctx) const;
	void postIntersection(unsigned idx, IntersectionContext * ctx) const;
	virtual char closestPoint(unsigned idx, const Vector3F & origin, IntersectionContext * ctx) const;
	char insideTriangle(const Vector3F & p, const Vector3F & a, const Vector3F & b, const Vector3F & c, const Vector3F & n) const;
	virtual unsigned closestVertex(unsigned idx, const Vector3F & px) const;
	char triangleIntersect(const Vector3F * threeCorners, IntersectionContext * ctx) const;
	char selectComponent(IntersectionContext * ctx) const;
	
	void putIntoObjectSpace();
	
	char hasVertexData(const std::string & name) const;
	float * perVertexFloat(const std::string & name);
	Vector3F * perVertexVector(const std::string & name);
	char * perFaceTag(const std::string & name);
	VertexDataGroup * vertexData();
	
	virtual unsigned processQuadFromPolygon();
	virtual unsigned * quadIndices();
	virtual unsigned * getQuadIndices() const;
	virtual unsigned numQuads() const;
	
	virtual const Type type() const;
	
	void verbose() const;
private:
	VertexDataGroup m_vdg;
	Vector3F * _vertices;
	Vector3F * m_normals;
	unsigned * _indices;
	unsigned m_numTriangles;
	unsigned m_numTriangleFaceVertices;
	unsigned m_numPolygons;
	unsigned * m_polygonCounts;
	unsigned m_numPolygonVertices;
	unsigned * m_polygonIndices;
	unsigned m_numUVs, m_numUVIds;
	float * m_u;
	float * m_v;
	unsigned * m_uvIds;
	unsigned _numVertices;
};

}
/*
 *  Facet.h
 *  convexHull
 *
 *  Created by jian zhang on 9/5/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <vector>
#include <math/Vector3F.h>
#include <topo/GeoElement.h>

namespace aphid {

class Vertex;
class Edge;

class Facet : public GeoElement {
public:
	Facet();
	Facet(Vertex *a, Vertex *b, Vertex *c);
	Facet(Vertex *a, Vertex *b, Vertex *c, Vector3F *d);
	virtual ~Facet();
	
	virtual void clear();
	virtual void setIndex(int val);
	char connectTo(Facet *another, Vertex *a, Vertex *b);
	Edge * matchedEdge(Vertex * a, Vertex * b);
	
	Edge * edge(int idx);
	Vertex * vertex(int idx);
	Vertex * vertexAfter(int idx);
	Vertex * vertexBefore(int idx);
	Vertex * thirdVertex(Vertex *a, Vertex *b);
	Vertex getVertex(int idx) const;
	Vector3F getCentroid() const;
	Vector3F getNormal() const;
	float getArea() const;
	
	char isVertexAbove(const Vertex & v) const;
	char isClosed() const;
	
	char getEdgeOnHorizon(std::vector<Edge *> & horizons) const;
	
	void update();
	void setPolygonIndex(unsigned idx);
	unsigned getPolygonIndex() const;

	static float cumputeArea(Vector3F *a, Vector3F *b, Vector3F *c);
private:
	void createEdges();
	Vector3F m_normal;
	float m_area;
	Vertex *m_vertices[3];
	Edge *m_edges[3];
	unsigned m_polyIndex;
};

}
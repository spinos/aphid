/*
 *  Edge.h
 *  convexHull
 *
 *  Created by jian zhang on 9/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Vertex.h>
#include <GeoElement.h>
class Edge: public GeoElement {
public:
	Edge();
	Edge(Vertex *a, Vertex *b, char * f);
	virtual ~Edge();
	
	char matches(Vertex *a, Vertex *b) const;
	void setTwin(Edge *e);
	Edge * getTwin() const;
	char * getFace() const;
	
	Vertex *v0();
	Vertex *v1();
	Vertex getV0() const;
	Vertex getV1() const;
	
	char isConnectedTo(Edge * another);
	void setNext(Edge * another);
	Edge* getNext() const;
	void flip();
	void disconnect();

private:
	Vertex *va;
	Vertex *vb;
	char *face;
	Edge *identicalTwin;
	Edge *next;
};
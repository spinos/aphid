/*
 *  AGraph.h
 *  
 *
 *  Graph system with nodes and edges
 *  per-node edge indices like 0 1 2 3 0 3 4 ...
 *  per-node edge offsets like 0 4 7 ...
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Types.h>
#include <Sequence.h>
#include <vector>
#include <iostream>

namespace aphid {

struct IGraphEdge {

	sdb::Coord2 vi;
	int index;
	float len;
	
};

template<typename Tn>
class AGraph {

	int m_numNode, m_numEdge, m_numEdgeInd;
	float m_minEdgeLen, m_maxEdgeLen;
	
	IGraphEdge * m_edges;
	int * m_vvEdgeIndices;
	int * m_vvEdgeBegins;
	Tn * m_nodes;
	
public:
	AGraph();
	virtual ~AGraph();
	
	void create(int nn, int ne, int ni);
	
	const int & numNodes() const;
	const int & numEdges() const;
	const int & numEdgeIndices() const;
	const float & minEdgeLength() const;
	const float & maxEdgeLength() const;
	
	Tn * nodes();
	IGraphEdge * edges();
	int * edgeIndices();
	int * edgeBegins();
	
	const Tn * nodes() const;
	const IGraphEdge * edges() const;
	const int * edgeIndices() const;
	const int * edgeBegins() const;
	
protected:
	void extractEdges(sdb::Sequence<sdb::Coord2> * a);
	void extractEdgeBegins(const std::vector<int> & a);
	void extractEdgeIndices(const std::vector<int> & a);
	void calculateEdgeLength();
	
private:
	void internalClear();

};

template<typename Tn>
AGraph<Tn>::AGraph() : 
m_numNode(0),
m_numEdge(0),
m_numEdgeInd(0),
m_edges(NULL),
m_vvEdgeIndices(NULL),
m_vvEdgeBegins(NULL),
m_nodes(NULL)
{}

template<typename Tn>
AGraph<Tn>::~AGraph()
{ internalClear(); }

template<typename Tn>
void AGraph<Tn>::internalClear()
{
	if(m_edges) delete[] m_edges;
	if(m_vvEdgeIndices) delete[] m_vvEdgeIndices;
	if(m_vvEdgeBegins) delete[] m_vvEdgeBegins;
	if(m_numNode > 0 && m_nodes) delete[] m_nodes;
}

template<typename Tn>
void AGraph<Tn>::create(int nn, int ne, int ni)
{
	internalClear();
	m_numNode = nn;
	m_numEdge = ne;
	m_numEdgeInd = ni;
	m_nodes = new Tn[m_numNode];
	m_edges = new IGraphEdge[m_numEdge];
	m_vvEdgeBegins = new int[m_numNode + 1];
	m_vvEdgeBegins[m_numNode] = m_numEdgeInd;
	m_vvEdgeIndices = new int[m_numEdgeInd];
}

template<typename Tn>
const int & AGraph<Tn>::numNodes() const
{ return m_numNode; }
	
template<typename Tn>
const int & AGraph<Tn>::numEdges() const
{ return m_numEdge; }

template<typename Tn>
const int & AGraph<Tn>::numEdgeIndices() const
{ return m_numEdgeInd; }

template<typename Tn>
Tn * AGraph<Tn>::nodes()
{ return m_nodes; }

template<typename Tn>
IGraphEdge * AGraph<Tn>::edges()
{ return m_edges; }

template<typename Tn>
int * AGraph<Tn>::edgeIndices()
{ return m_vvEdgeIndices; }

template<typename Tn>
int * AGraph<Tn>::edgeBegins()
{ return m_vvEdgeBegins; }

template<typename Tn>
const Tn * AGraph<Tn>::nodes() const
{ return m_nodes; }

template<typename Tn>
const IGraphEdge * AGraph<Tn>::edges() const
{ return m_edges; }

template<typename Tn>
const int * AGraph<Tn>::edgeIndices() const
{ return m_vvEdgeIndices; }

template<typename Tn>
const int * AGraph<Tn>::edgeBegins() const
{ return m_vvEdgeBegins; }

template<typename Tn>
void AGraph<Tn>::extractEdges(sdb::Sequence<sdb::Coord2> * a)
{
	int i = 0;
	a->begin();
	while(!a->end() ) {
		
		IGraphEdge * e = &m_edges[i];
		e->vi = a->key();
		e->index = i;
		
		i++;
		a->next();
	}
}

template<typename Tn>
void AGraph<Tn>::extractEdgeBegins(const std::vector<int> & a)
{
	int i = 0;
	std::vector<int>::const_iterator it = a.begin();
	for(;it!=a.end();++it) {
		m_vvEdgeBegins[i++] = *it;
	}
}
 
template<typename Tn>
void AGraph<Tn>::extractEdgeIndices(const std::vector<int> & a)
{
	int i = 0;
	std::vector<int>::const_iterator it = a.begin();
	for(;it!=a.end();++it) {
		m_vvEdgeIndices[i++] = *it;
	}
}

template<typename Tn>
void AGraph<Tn>::calculateEdgeLength()
{
	m_minEdgeLen = 1e9f;
	m_maxEdgeLen = -1e9f;
	
	const int n = numEdges();
	int i = 0;
	for(;i<n;++i) {
		
		IGraphEdge & ei = m_edges[i];
		ei.len = m_nodes[ei.vi.x].pos.distanceTo(m_nodes[ei.vi.y].pos);
		
		if(m_minEdgeLen > ei.len)
			m_minEdgeLen = ei.len;
			
		if(m_maxEdgeLen < ei.len)
			m_maxEdgeLen = ei.len;
	}
}

template<typename Tn>
const float & AGraph<Tn>::minEdgeLength() const
{ return m_minEdgeLen; }

template<typename Tn>
const float & AGraph<Tn>::maxEdgeLength() const
{ return m_maxEdgeLen; }	
	
}
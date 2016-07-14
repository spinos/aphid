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

namespace aphid {

struct IGraphEdge {

	sdb::Coord2 vi;
	int index;
	
};

template<typename Tn>
class AGraph {

	int m_numNode, m_numEdge;
	
	IGraphEdge * m_edges;
	int * m_vvEdgeIndices;
	int * m_vvEdgeBegins;
	Tn * m_nodes;
	
public:
	AGraph();
	virtual ~AGraph();
	
	void clear();
	void create(int nn, int ne, int ni);
	
	const int & numNodes() const;
	const int & numEdges() const;
	
	Tn * nodes();
	IGraphEdge * edges();
	int * edgeIndices();
	int * edgeBegins();
	
protected:

private:


};

template<typename Tn>
AGraph<Tn>::AGraph() : 
m_numNode(0),
m_numEdge(0),
m_edges(NULL),
m_vvEdgeIndices(NULL),
m_vvEdgeBegins(NULL),
m_nodes(NULL)
{}

template<typename Tn>
AGraph<Tn>::~AGraph()
{ clear(); }

template<typename Tn>
void AGraph<Tn>::clear()
{
	if(m_edges) delete[] m_edges;
	if(m_vvEdgeIndices) delete[] m_vvEdgeIndices;
	if(m_vvEdgeBegins) delete[] m_vvEdgeBegins;
	if(m_nodes) delete[] m_nodes;
}

template<typename Tn>
void AGraph<Tn>::create(int nn, int ne, int ni)
{
	clear();
	m_numNode = nn;
	m_numEdge = ne;
	m_nodes = new Tn[m_numNode];
	m_edges = new IGraphEdge[m_numEdge];
	m_vvEdgeBegins = new int[m_numNode + 1];
	m_vvEdgeBegins[m_numNode] = ni;
	m_vvEdgeIndices = new int[ni];
}

template<typename Tn>
const int & AGraph<Tn>::numNodes() const
{ return m_numNode; }
	
template<typename Tn>
const int & AGraph<Tn>::numEdges() const
{ return m_numEdge; }

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

}
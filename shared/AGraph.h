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

template<typename Tn, typename Te>
class AGraph {

	int m_numNode, m_numEdge, m_numEdgeInd;
	float m_minEdgeLen, m_maxEdgeLen;
	
	Te * m_edges;
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
	Te * edges();
	int * edgeIndices();
	int * edgeBegins();
	
	const Tn * nodes() const;
	const Te * edges() const;
	const int * edgeIndices() const;
	const int * edgeBegins() const;
	
	const Te * edge(const int & v1, const int & v2) const;
	
	void verbose() const;

protected:
	void extractEdges(sdb::Sequence<sdb::Coord2> * a);
	void extractEdgeBegins(const std::vector<int> & a);
	void extractEdgeIndices(const std::vector<int> & a);
	void calculateEdgeLength();
/// ind to edge by vertex i
	int edgeIndex(const int & v1, const int & v2) const;
	
private:
	void internalClear();

};

template<typename Tn, typename Te>
AGraph<Tn, Te>::AGraph() : 
m_numNode(0),
m_numEdge(0),
m_numEdgeInd(0),
m_edges(NULL),
m_vvEdgeIndices(NULL),
m_vvEdgeBegins(NULL),
m_nodes(NULL)
{}

template<typename Tn, typename Te>
AGraph<Tn, Te>::~AGraph()
{ internalClear(); }

template<typename Tn, typename Te>
void AGraph<Tn, Te>::internalClear()
{
	if(m_edges) delete[] m_edges;
	if(m_vvEdgeIndices) delete[] m_vvEdgeIndices;
	if(m_vvEdgeBegins) delete[] m_vvEdgeBegins;
	if(m_numNode > 0 && m_nodes) delete[] m_nodes;
}

template<typename Tn, typename Te>
void AGraph<Tn, Te>::create(int nn, int ne, int ni)
{
	internalClear();
	m_numNode = nn;
	m_numEdge = ne;
	m_numEdgeInd = ni;
	m_nodes = new Tn[m_numNode];
	m_edges = new Te[m_numEdge];
	m_vvEdgeBegins = new int[m_numNode + 1];
	m_vvEdgeBegins[m_numNode] = m_numEdgeInd;
	m_vvEdgeIndices = new int[m_numEdgeInd];
}

template<typename Tn, typename Te>
const int & AGraph<Tn, Te>::numNodes() const
{ return m_numNode; }
	
template<typename Tn, typename Te>
const int & AGraph<Tn, Te>::numEdges() const
{ return m_numEdge; }

template<typename Tn, typename Te>
const int & AGraph<Tn, Te>::numEdgeIndices() const
{ return m_numEdgeInd; }

template<typename Tn, typename Te>
Tn * AGraph<Tn, Te>::nodes()
{ return m_nodes; }

template<typename Tn, typename Te>
Te * AGraph<Tn, Te>::edges()
{ return m_edges; }

template<typename Tn, typename Te>
int * AGraph<Tn, Te>::edgeIndices()
{ return m_vvEdgeIndices; }

template<typename Tn, typename Te>
int * AGraph<Tn, Te>::edgeBegins()
{ return m_vvEdgeBegins; }

template<typename Tn, typename Te>
const Tn * AGraph<Tn, Te>::nodes() const
{ return m_nodes; }

template<typename Tn, typename Te>
const Te * AGraph<Tn, Te>::edges() const
{ return m_edges; }

template<typename Tn, typename Te>
const int * AGraph<Tn, Te>::edgeIndices() const
{ return m_vvEdgeIndices; }

template<typename Tn, typename Te>
const int * AGraph<Tn, Te>::edgeBegins() const
{ return m_vvEdgeBegins; }

template<typename Tn, typename Te>
void AGraph<Tn, Te>::extractEdges(sdb::Sequence<sdb::Coord2> * a)
{
	int i = 0;
	a->begin();
	while(!a->end() ) {
		
		Te * e = &m_edges[i];
		e->vi = a->key();
		e->err = 0.f;
		e->cx = -1.f;
		
		i++;
		a->next();
	}
}

template<typename Tn, typename Te>
void AGraph<Tn, Te>::extractEdgeBegins(const std::vector<int> & a)
{
	int i = 0;
	std::vector<int>::const_iterator it = a.begin();
	for(;it!=a.end();++it) {
		m_vvEdgeBegins[i++] = *it;
	}
}
 
template<typename Tn, typename Te>
void AGraph<Tn, Te>::extractEdgeIndices(const std::vector<int> & a)
{
	int i = 0;
	std::vector<int>::const_iterator it = a.begin();
	for(;it!=a.end();++it) {
		m_vvEdgeIndices[i++] = *it;
	}
}

template<typename Tn, typename Te>
void AGraph<Tn, Te>::calculateEdgeLength()
{
	m_minEdgeLen = 1e9f;
	m_maxEdgeLen = -1e9f;
	
	const int n = numEdges();
	int i = 0;
	for(;i<n;++i) {
		
		Te & ei = m_edges[i];
		ei.len = m_nodes[ei.vi.x].pos.distanceTo(m_nodes[ei.vi.y].pos);
		
		if(m_minEdgeLen > ei.len)
			m_minEdgeLen = ei.len;
			
		if(m_maxEdgeLen < ei.len)
			m_maxEdgeLen = ei.len;
	}
}

template<typename Tn, typename Te>
const float & AGraph<Tn, Te>::minEdgeLength() const
{ return m_minEdgeLen; }

template<typename Tn, typename Te>
const float & AGraph<Tn, Te>::maxEdgeLength() const
{ return m_maxEdgeLen; }	
	
template<typename Tn, typename Te>
int AGraph<Tn, Te>::edgeIndex(const int & v1, const int & v2) const
{
	if(numEdgeIndices() < 1) return -1;
	if(v1>= numNodes() ) {
		std::cout<<"\n AGraph invalid node ind "<<v1;
		return -1;
	}
	if(v2>= numNodes() ) {
		std::cout<<"\n AGraph invalid node ind "<<v2;
		return -1;
	}
	
	const Tn & A = nodes()[v1];
	const int endj = edgeBegins()[v1+1];
	int j = edgeBegins()[v1];
	for(;j<endj;++j) {
		
		if(j>numEdgeIndices()-1 ) {
			std::cout<<"\n AGraph invalid edge ind "<<j;
		}
		
		int k = edgeIndices()[j];
		
		if(k>numEdges()-1 ) {
			std::cout<<"\n AGraph invalid edge i "<<k;
		}

		const Te & eg = edges()[k];
		
		if(eg.vi.x == v2 || eg.vi.y == v2)
			return k;
	}
	return -1;
}

template<typename Tn, typename Te>
const Te * AGraph<Tn, Te>::edge(const int & v1, const int & v2) const
{ 
	int k = edgeIndex(v1, v2); 
	if(k<0) {
		//std::cout<<"\n invalid edge ("<<v1<<","<<v2<<")"<<std::endl;
		return NULL;
	}
		
	return &edges()[k];
}

template<typename Tn, typename Te>
void AGraph<Tn, Te>::verbose() const
{
	std::cout<<"\n graph n node "<<numNodes()
			<<"    n edge "<<numEdges()
			<<"    n ind "<<numEdgeIndices()
			<<"    min/max edge len "<<minEdgeLength()
								<<"/"<<maxEdgeLength();
}

}
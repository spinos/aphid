/*
 *  SVOTraverser.h
 *  
 *
 *  Created by jian zhang on 2/15/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_SVO_TRAVERSER_H
#define APH_TTG_SVO_TRAVERSER_H

#include <iostream>
#include <boost/scoped_array.hpp>
#include <deque>

namespace aphid {

namespace ttg {

/// for traversing
struct SVOTNode {

/// (child[0,7], parent, location)
	int _ind[16];
	
};

template<typename T>
class SVOTraverser {

	boost::scoped_array<T> m_nodes;
	int m_numNodes;
/// (center, half_span) of root node
	float m_coord[4];
	
public:
	
	SVOTraverser();
	~SVOTraverser();
	
	float* coord();
	const float* coord() const;
	
	T& node(int i);
	const T& node(int i) const;
	
	const T* nodes() const;
	
	void createNumNodes(int n);
	
	const int& numNodes() const;
	
protected:

private:

};

template<typename T>
SVOTraverser<T>::SVOTraverser() : m_numNodes(0)
{}

template<typename T>
SVOTraverser<T>::~SVOTraverser()
{}

template<typename T>
void SVOTraverser<T>::createNumNodes(int n)
{ 
	m_numNodes = n;
	m_nodes.reset(new T[n]); 
}

template<typename T>
T& SVOTraverser<T>::node(int i)
{ return m_nodes.get()[i]; }

template<typename T>
const T& SVOTraverser<T>::node(int i) const
{ return m_nodes.get()[i]; }

template<typename T>
float* SVOTraverser<T>::coord()
{ return m_coord; }

template<typename T>
const float* SVOTraverser<T>::coord() const
{ return m_coord; }

template<typename T>
const T* SVOTraverser<T>::nodes() const
{ return m_nodes.get(); }

template<typename T>
const int& SVOTraverser<T>::numNodes() const
{ return m_numNodes; }

template<typename T, typename Tr>
class StackedDrawContext {

	struct DrawEvent {
		T _node;
		float _coord[4];
	};
	
	std::deque<DrawEvent> m_drawQueue;
	Tr* m_rule;
	const T* m_nodes;
	
public:

	StackedDrawContext(Tr* rule);
	
	void begin(const T* nodes, const float* coord);
	bool end();
	void next();
	
	const float* currentCoord() const;
	
};

template<typename T, typename Tr>
StackedDrawContext<T, Tr>::StackedDrawContext(Tr* rule)
{ m_rule = rule; }

template<typename T, typename Tr>
void StackedDrawContext<T, Tr>::begin(const T* nodes, const float* coord)
{
	m_nodes = nodes;
	DrawEvent e;
	e._node = m_nodes[0];
	memcpy(e._coord, coord, 16);
	
	m_drawQueue.push_back(e);
}

template<typename T, typename Tr>
bool StackedDrawContext<T, Tr>::end()
{ return m_drawQueue.size() < 1; }

template<typename T, typename Tr>
void StackedDrawContext<T, Tr>::next()
{
	const DrawEvent& fe = m_drawQueue.front();
	const T& fn = fe._node;
	//std::cout<<"\n p ";
	//m_rule->printCoord(fe._coord);
	
	DrawEvent subdraw;
	for(int i=0;i<8;++i) {
		
		if(fn._ind[i] < 1)
			continue;
		
		subdraw._node = m_nodes[fn._ind[i] ];
		m_rule->computeChildCoord(subdraw._coord, fe._coord, i);
		//std::cout<<"\n c"<<i;
		//m_rule->printCoord(subdraw._coord);
		m_drawQueue.push_back(subdraw);
		
	}
	
	m_drawQueue.pop_front();
}

template<typename T, typename Tr>
const float* StackedDrawContext<T, Tr>::currentCoord() const
{ 
	const DrawEvent& fe = m_drawQueue.front();
	return fe._coord;
}

}

}

#endif

/*
 *  SparseVoxelOctree.h
 *  
 *  T is node type Ts sample type
 *
 *  Created by jian zhang on 2/13/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_SPARSE_VOXEL_OCTREE_H
#define APH_TTG_SPARSE_VOXEL_OCTREE_H

#include <sdb/SpaceFillingVector.h>
#include "SVOTraverser.h"

namespace aphid {

namespace ttg {

/// for building 
struct SVOBNode {

/// sfc
	int _key;
/// end of sfc	
	int _nextKey;
	int _level;
/// (to_parent, to_child[0,7], location)
/// to_parent <- (child_index | to_parent)
	int _ind[10];
/// build box (center, half_sapn)
	float _coord[4];
/// range of samples
	int _range[2];
	
	void setChildKey(const int& x, const int& i);
	
	void setLeaf();
	
	bool isLeaf() const;
	
	void connectToParent(const int& pid, const int& i);
/// to_parent decoded	
	int parentInd() const;
	
	void getChildInds(int* dst) const;
	
	void getParent(int& dst) const;

	void getLocation(int& dst) const;

};

template<typename T>
class SVOBuilder {

	sdb::SpaceFillingVector<T> m_nodes;
/// (center, half_span) of root node
	float m_coord[4];
	
public:

	SVOBuilder();
	~SVOBuilder();
	
	template<typename Ts, typename Tr>
	void build(const sdb::SpaceFillingVector<Ts>& samples, 
				Tr& rule);
				
	template<typename Tt, typename Tr>
	void save(SVOTraverser<Tt>& traverser, 
				Tr& rule) const;
	
protected:

private:

	template<typename Ts, typename Tr>
	void subdivideNode(std::deque<T>& buildQueue, 
				T& parentNode,
				const sdb::SpaceFillingVector<Ts>& samples, 
				Tr& rule);

	template<typename Ts, typename Tr>
	void buildFirstNodeInQueue(std::deque<T>& buildQueue, 
				const sdb::SpaceFillingVector<Ts>& samples, 
				Tr& rule);
				
	template<typename Tr>
	void connectNodes(T& parentNode,
				const int& parentInd,
				Tr& rule);
	
	bool checkConnections() const;
	
};

template<typename T>
SVOBuilder<T>::SVOBuilder()
{}

template<typename T>
SVOBuilder<T>::~SVOBuilder()
{ m_nodes.clear(); }

template<typename T>
template<typename Ts, typename Tr>
void SVOBuilder<T>::build(const sdb::SpaceFillingVector<Ts>& samples, 
				Tr& rule)
{
	T rootNode;
	rule. template buildRoot<T, Ts>(rootNode, m_coord, samples);
		
/// stack waiting to build
	std::deque<T> buildQueue;
	
	buildQueue.push_back(rootNode);
	
	while(!buildQueue.empty() ) {
		
		buildFirstNodeInQueue<Ts, Tr>(buildQueue, samples, rule);
	}
	
	const int n = m_nodes.size();
	std::cout<<"\n n svo nodes "<<n;
	for(int i=0;i<n;++i) {
		m_nodes[i]._key = rule.computeFinestKey(m_nodes[i]._coord);
		m_nodes[i]._ind[9] = i;
	}
	
	m_nodes[0]._key = -1;
	
	m_nodes.sort();
	
	for(int i=0;i<n;++i) {
		connectNodes(m_nodes[i], i, rule);
	}
	
#if 0
	for(int i=0;i<n;++i) {
		std::cout<<"\n n "<<i;
		rule. template printNode<T>(m_nodes[i]);
	}
#endif
	
	if(checkConnections()) {
		std::cout<<"\n passed check ";
	}
	
}

template<typename T>
bool SVOBuilder<T>::checkConnections() const
{
	const int n = m_nodes.size();
	for(int i=1;i<n;++i) {
		const T& ni = m_nodes[i];
		if(ni._ind[0] < 0) {
			std::cout<<"\n node "<<i<<" is not connected ";
			return false;
		}
		if(ni._level == 1 && ni.parentInd() != 0) {
			std::cout<<"\n l1 node "<<i<<" is not connected to l0";
			return false;
		}
		
	}
	return true;
}

template<typename T>
template<typename Ts, typename Tr>
void SVOBuilder<T>::subdivideNode(std::deque<T>& buildQueue, 
				T& parentNode,
				const sdb::SpaceFillingVector<Ts>& samples, 
				Tr& rule)
{
	T subNode[8];
	rule. template computeChildKeys(subNode, parentNode);
	
	for(int i=0;i<8;++i) {
	
		int state = rule. template spawnNode<T, Ts>(subNode, i, parentNode, samples);
		if(!state) {
			parentNode.setChildKey(0, i);
			continue;
		}
		
		parentNode.setChildKey(subNode[i]._key, i);		
		
		buildQueue.push_back(subNode[i]);
		
	}
}

template<typename T>
template<typename Ts, typename Tr>
void SVOBuilder<T>::buildFirstNodeInQueue(std::deque<T>& buildQueue, 
				const sdb::SpaceFillingVector<Ts>& samples, 
				Tr& rule)
{
	T& node = buildQueue.front();
	if(node._level > 1)
		rule. template buildInner<T, Ts>(node, samples);
	
	if(rule.endSubdivide(node) ) {
		node.setLeaf();
	} else {	
		subdivideNode<Ts, Tr>(buildQueue, node, samples, rule);
		
	}
	node._ind[0] = -1;
	m_nodes.push_back(node);
	
	buildQueue.pop_front();
}

template<typename T>
template<typename Tr>
void SVOBuilder<T>::connectNodes(T& parentNode,
				const int& parentInd,
				Tr& rule)
{
	if(parentNode.isLeaf() )
		return;
		
	float childCoord[4];
	for(int i=0;i<8;++i) {
		rule.computeChildCoord(childCoord, parentNode._coord, i);
		int k = rule.computeFinestKey(childCoord);
			
		int childInd = m_nodes.findElement(k);
		
		if(childInd < 0) {
			parentNode.setChildKey(0, i);
			
		} else {
			parentNode.setChildKey(childInd, i);
			T& ci = m_nodes[childInd];
			ci.connectToParent(parentInd, i);
		}
	}
}

template<typename T>
template<typename Tt, typename Tr>
void SVOBuilder<T>::save(SVOTraverser<Tt>& traverser, 
				Tr& rule) const
{
	const int n = m_nodes.size();
	traverser.createNumNodes(n);
	
	rule. template saveRootNode<Tt, T>(traverser.node(0),
								traverser.coord(),
								m_nodes[0]);
								
	for(int i=1;i<n;++i) {
		rule. template saveNode<Tt, T>(traverser.node(i),
								m_nodes[i]);
	}
}

}

}

#endif

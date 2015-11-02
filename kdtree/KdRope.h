/*
 *  KdRope.h
 *  testntree
 *
 *  Created by jian zhang on 11/2/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "Treelet.h"
#include "KdNTree.h"

template<int NumLevels, typename T, typename Tn>
class KdRope : public Treelet<NumLevels > {
	
	BoundingBox m_boxes[(1<<NumLevels+1) - 2];
	int m_splitAxis[(1<<NumLevels+1) - 2];
	float m_splitPos[(1<<NumLevels+1) - 2];
	KdNeighbors m_ns[(1<<NumLevels+1) - 2];
	KdNTree<T, Tn> * m_tree;
	static std::map<unsigned, BoundingBox > BoxMap;
	
public:
	KdRope(int index, KdNTree<T, Tn> * tree);
	virtual ~KdRope() {}
	
	void build(int parentTreelet, int parentNodeIdx, const BoundingBox & box, const KdNeighbors & ns);
	
	BoundingBox box(int idx) const;
	KdNeighbors neighbor(int idx) const;
	
	void beginMap();
	void endMap();
	
protected:
	void visitRoot(KdTreeNode * parent, const BoundingBox & box, const KdNeighbors & ns);
	bool visitInterial(int level);
	void visitCousins(int iNode, int level);
	bool chooseCousinAsNeighbor(int iNeighbor, int iNode, int iParent, int & updated);
	void pushLeaves();
	void mapNeighbors(KdNeighbors & ns);
private:
	
};

template<int NumLevels, typename T, typename Tn>
std::map<unsigned, BoundingBox > KdRope<NumLevels, T, Tn>::BoxMap;

template<int NumLevels, typename T, typename Tn>
KdRope<NumLevels, T, Tn>::KdRope(int index, KdNTree<T, Tn> * tree) : Treelet<NumLevels>(index)
{
	m_tree = tree;
	const int n = Treelet<NumLevels>::numNodes();
	int i = 0;
	for(;i<n;i++) m_ns[i].reset();
}

template<int NumLevels, typename T, typename Tn>
BoundingBox KdRope<NumLevels, T, Tn>::box(int idx) const
{ return m_boxes[idx]; }

template<int NumLevels, typename T, typename Tn>
KdNeighbors KdRope<NumLevels, T, Tn>::neighbor(int idx) const
{ return m_ns[idx]; }

template<int NumLevels, typename T, typename Tn>
void KdRope<NumLevels, T, Tn>::build(int parentTreelet, int parentNodeIdx, const BoundingBox & box, const KdNeighbors & ns)
{
	Tn * root = &m_tree->nodes()[parentTreelet];
	KdTreeNode * rootNode = root->node(parentNodeIdx);
	visitRoot(rootNode, box, ns);
	int level = 1;
    for(;level <= NumLevels; level++) {
		if(!visitInterial(level)) break;
	}
	pushLeaves();
}

template<int NumLevels, typename T, typename Tn>
void KdRope<NumLevels, T, Tn>::visitRoot(KdTreeNode * parent, const BoundingBox & box, const KdNeighbors & ns)
{
	const float pos = parent->getSplitPos();
	const int axis = parent->getAxis();
	BoundingBox lftBox, rgtBox;
	box.split(axis, pos, lftBox, rgtBox);
	m_boxes[0] = lftBox;
	m_boxes[1] = rgtBox;
	m_ns[0] = ns;
	m_ns[1] = ns;
	/// opposite sides of the split become neighbors along split axis
	m_ns[0].set(rgtBox, axis, true, Treelet<NumLevels>::index(), 1);
	m_ns[1].set(lftBox, axis, false, Treelet<NumLevels>::index(), 0);
}

template<int NumLevels, typename T, typename Tn>
bool KdRope<NumLevels, T, Tn>::visitInterial(int level)
{
	bool needNextLevel = false;
	const int levelBegin = Treelet<NumLevels>::OffsetByLevel(level);
	const int iTreelet = Treelet<NumLevels>::index();
	Tn * treelet = &m_tree->nodes()[iTreelet];
	const int nAtLevel = 1<<level;
	BoundingBox lftBox, rgtBox;
	int i;
	if(level > 1) {
		for(i=0; i<nAtLevel; i++) {
			const int iNode = levelBegin + i;
			KdTreeNode * node = treelet->node(iNode);
			visitCousins(iNode, level);
		}
	}
    
	if(level < NumLevels) {
		for(i=0; i<nAtLevel; i++) {
			const int iNode = levelBegin + i;
			KdTreeNode * node = treelet->node(iNode);
			if(node->isLeaf()) continue;

			const BoundingBox nodeBox = m_boxes[iNode];
			const float pos = node->getSplitPos();
			const int axis = node->getAxis();
			
			m_splitAxis[iNode] = axis;
			m_splitPos[iNode] = pos;
		
			nodeBox.split(axis, pos, lftBox, rgtBox);
			
			const int iLftChild = iNode + Treelet<NumLevels>::ChildOffset(iNode);
			m_boxes[iLftChild] = lftBox;
			m_boxes[iLftChild + 1] = rgtBox;
			
			const KdNeighbors nodeNs = m_ns[iNode];
			m_ns[iLftChild] = nodeNs;
			m_ns[iLftChild + 1] = nodeNs;
			
			m_ns[iLftChild].set(rgtBox, axis, true, iTreelet, iLftChild + 1);
			m_ns[iLftChild + 1].set(lftBox, axis, false, iTreelet, iLftChild);
			
			needNextLevel = true;
		}
	}
	
	return needNextLevel;
}

template<int NumLevels, typename T, typename Tn>
void KdRope<NumLevels, T, Tn>::visitCousins(int iNode, int level)
{
	const int hi = Treelet<NumLevels>::OffsetByLevel(level);
	KdNeighbors & nodeNs = m_ns[iNode];
	int i = 0;
	for(;i<6;i++) {
		BoundingBox & ni = nodeNs._n[i];
		if(ni.m_padding1 != 0) {
			if(ni.m_padding0 < hi) {
				int updated;
				if(chooseCousinAsNeighbor(i, iNode, ni.m_padding0, updated)) {
					// std::cout<<"\n ud nei "<<iNode<<" "<<ni.m_padding0<<" -> "<<updated;
					m_ns[iNode].set(m_boxes[updated], i, Treelet<NumLevels>::index(), updated);
				}
			}
		}
	}
}

template<int NumLevels, typename T, typename Tn>
bool KdRope<NumLevels, T, Tn>::chooseCousinAsNeighbor(int iNeighbor, int iNode, int iParent, int & updated)
{
	const BoundingBox a = m_boxes[iNode];
	const int iLftCousin = iParent + Treelet<NumLevels>::ChildOffset(iParent);
	BoundingBox b = m_boxes[iLftCousin];
	if(KdNeighbors::IsNeighborOf(iNeighbor, a, b)) {
		updated = iLftCousin;
		return true;
	}
	b = m_boxes[iLftCousin + 1];
	if(KdNeighbors::IsNeighborOf(iNeighbor, a, b)) {
		updated = iLftCousin + 1;
		return true;
	}
	return false;
}

template<int NumLevels, typename T, typename Tn>
void KdRope<NumLevels, T, Tn>::pushLeaves()
{
	const int iTreelet = Treelet<NumLevels>::index();
	Tn * treelet = &m_tree->nodes()[iTreelet];
	const int n = Treelet<NumLevels>::numNodes();
	int i = 0;
	for(;i<n;i++) {
		KdNeighbors nodeNs = m_ns[i];
		if(nodeNs.isEmpty()) continue;
		
		KdTreeNode * node = treelet->node(i);
		if(node->isLeaf()) {
			// std::cout<<" rope treelet["<<iTreelet<<"] leaf["<<i<<"]";
			m_tree->setLeafRope(node->getPrimStart(), nodeNs);
			mapNeighbors(nodeNs);
		}
	}
}

template<int NumLevels, typename T, typename Tn>
void KdRope<NumLevels, T, Tn>::beginMap()
{ BoxMap.clear(); }

template<int NumLevels, typename T, typename Tn>
void KdRope<NumLevels, T, Tn>::mapNeighbors(KdNeighbors & ns)
{
	int i = 0;
	for(;i<6;i++) {
		if(ns._n[i].m_padding1 != 0) {
			unsigned k = ns.encodeTreeletNodeHash(i, Tn::BranchingFactor);
			ns._n[i].m_padding1 = k;
			BoxMap[k] = ns._n[i];
		}
	}
}

template<int NumLevels, typename T, typename Tn>
void KdRope<NumLevels, T, Tn>::endMap()
{
	std::cout<<"\n n ropes "<<BoxMap.size();
	unsigned i = 0;
	std::map<unsigned, BoundingBox >::iterator it = BoxMap.begin();
	for(;it!=BoxMap.end(); ++it) {
		it->second.m_padding0 = i;
		i++;
	}
	
	m_tree->createRopes(BoxMap.size());
	i = 0;
	it = BoxMap.begin();
	for(;it!=BoxMap.end(); ++it) {
		m_tree->setRope(i, it->second);
		i++;
	}
	
	const unsigned n = m_tree->numLeafNodes();
	int j;
	i = 0;
	for(;i<n;i++) {
		for(j=0;j<6;j++) {
			unsigned k = m_tree->leafRopeInd(i, j);
			if(k != 0) {
				m_tree->setLeafRopeInd(BoxMap[k].m_padding0, i, j);
			}
		}
	}
}
//:~
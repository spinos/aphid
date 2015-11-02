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

struct KdNeighbors {
	/// 0 left 1 right 2 bottom 3 top 4 back 5 front
	
	BoundingBox _n[6];
	void reset() 
	{
		int i = 0;
		for(;i<6;i++) {
			_n[i].m_padding0 = 0; 
			_n[i].m_padding1 = 0; // zero is null
		}
	}
	
	void set(const BoundingBox & box, int axis, bool isHigh, int treeletIdx, int nodeIdx)
	{
		int idx = axis<<1;
		if(isHigh) idx++;
		_n[idx] = box;
		_n[idx].m_padding0 = nodeIdx;
		_n[idx].m_padding1 = treeletIdx;
	}
	
	bool isEmpty() const
	{
		int i = 0;
		for(;i<6;i++) {
			if(_n[i].m_padding1 != 0) return false;
		}
		return true;
	}
	
	void verbose() const
	{
		int i = 0;
		for(;i<6;i++) {
			if(_n[i].m_padding1 != 0) std::cout<<"\n ["<<i<<"] "<<_n[i].m_padding0
				<<" "<<_n[i].m_padding1
				<<" "<<_n[i];
		}
	}
};

template<int NumLevels, typename T, typename Tn>
class KdRope : public Treelet<NumLevels > {
	
	BoundingBox m_boxes[(1<<NumLevels+1) - 2];
	int m_splitAxis[(1<<NumLevels+1) - 2];
	float m_splitPos[(1<<NumLevels+1) - 2];
	KdNeighbors m_ns[(1<<NumLevels+1) - 2];
	KdNTree<T, Tn> * m_tree;
	
public:
	KdRope(int index, KdNTree<T, Tn> * tree);
	virtual ~KdRope() {}
	
	void build(int parentTreelet, int parentNodeIdx, const BoundingBox & box, const KdNeighbors & ns);
	
protected:
	void visitRoot(KdTreeNode * parent, const BoundingBox & box, const KdNeighbors & ns);
	bool visitInterial(int level);
	void pushLeaves();
private:
	
};

template<int NumLevels, typename T, typename Tn>
KdRope<NumLevels, T, Tn>::KdRope(int index, KdNTree<T, Tn> * tree) : Treelet<NumLevels>(index)
{
	m_tree = tree;
	const int n = Treelet<NumLevels>::numNodes();
	int i = 0;
	for(;i<n;i++) m_ns[i].reset();
}

template<int NumLevels, typename T, typename Tn>
void KdRope<NumLevels, T, Tn>::build(int parentTreelet, int parentNodeIdx, const BoundingBox & box, const KdNeighbors & ns)
{
	Tn * root = &m_tree->nodes()[parentTreelet];
	KdTreeNode * rootNode = root->node(parentNodeIdx);
	visitRoot(rootNode, box, ns);
	int level = 1;
    for(;level < NumLevels; level++) {
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
	const int iTreelet = Treelet<NumLevels>::index();
	Tn * treelet = &m_tree->nodes()[iTreelet];
	const int nAtLevel = 1<<level;
	BoundingBox lftBox, rgtBox;
    int i;
    for(i=0; i<nAtLevel; i++) {
		const int iNode = Treelet<NumLevels>::OffsetByLevel(level) + i;
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
	return needNextLevel;
}

template<int NumLevels, typename T, typename Tn>
void KdRope<NumLevels, T, Tn>::pushLeaves()
{
	const int iTreelet = Treelet<NumLevels>::index();
	Tn * treelet = &m_tree->nodes()[iTreelet];
	const int n = Treelet<NumLevels>::numNodes();
	int i = 0;
	for(;i<n;i++) {
		const KdNeighbors nodeNs = m_ns[i];
		if(nodeNs.isEmpty()) continue;
		
		KdTreeNode * node = treelet->node(i);
		if(node->isLeaf()) {
			std::cout<<"\n leaf "<<i<<" nei";
			nodeNs.verbose();
		}
	}
}
//:~
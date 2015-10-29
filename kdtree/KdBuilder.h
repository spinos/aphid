/*
 *  KdBuilder.h
 *  aphid
 *
 *  Created by jian zhang on 10/29/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "KdSah.h"

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
class KdTreelet {
	static int NumPrimsInLeaf;
    static int NumSubSplits;
	///
    ///       [1            2]0
    ///   [3    4]1      [5      6]2
    ///  [7 8]3 [9 10]4  [11 12]5 [13 14]6
    ///
    SahSplit<T> * m_splits[(1<<NumLevels+1) - 1];
	Tn * m_node;
public:
	KdTreelet();
	virtual ~KdTreelet();
	
	void build(SahSplit<T> * parent, Tn * dst);
	
protected:
	bool subdivide1(int level);
	void createLeaf(SahSplit<T> * parent, int level);
	void clearSplit(int idx, int level);
	void setNodeInternal(int idx, int axis, float pos);
	void setNodeLeaf(int idx);
private:

};

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
int KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::NumPrimsInLeaf = 1<<NumPrimsInLeafLog;

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
int KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::NumSubSplits = (1<<NumLevels+1) - 1;

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::KdTreelet()
{
	int i;
	for(i=0;i<NumSubSplits;i++) {
		m_splits[i] = NULL;
	}
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::~KdTreelet()
{
	int i;
	for(i=1;i<NumSubSplits;i++) {
		if(m_splits[i]) delete m_splits[i];
	}
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::build(SahSplit<T> * parent, Tn * dst)
{
	m_node = dst;
    m_splits[0] = parent;
    int level = 0;
    for(;level < NumLevels; level++) {
        if(!subdivide1(level)) break;
	}
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
bool KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::subdivide1(int level)
{
	bool needNextLevel = false;
    std::cout<<"\n\n subdiv level "<<level;
    const int nSplitAtLevel = 1<<level;
    int i;
    for(i=0; i<nSplitAtLevel; i++) {
        const int iNode = Tn::LevelOffset[level] + i;
        const int iLftChild = iNode + iNode + 1;
        std::cout<<"\n split node "<<iNode;
        // std::cout<<" child offset "<<iLftChild;
        
        SahSplit<T>  * parent = m_splits[iNode];
		
		if(parent->numPrims() <= NumPrimsInLeaf) {
			createLeaf(parent, level);
			setNodeLeaf(iNode);
			clearSplit(iNode, level);
			continue;
		}
	
        SplitEvent * plane = parent->bestSplit();
		
		if(plane->getCost() > parent->visitCost()) {
			std::cout<<"\n visit cost "
				<<parent->visitCost()
				<<" < split cost "
				<<plane->getCost()
				<<" stop subdivide\n";
			createLeaf(parent, level);
			setNodeLeaf(iNode);
			clearSplit(iNode, level);
			continue;
		}
		
		SahSplit<T>  * lftChild = new SahSplit<T>(plane->leftCount());
		SahSplit<T>  * rgtChild = new SahSplit<T>(plane->rightCount());
		parent->partition(lftChild, rgtChild);
		
		m_splits[iLftChild] = lftChild;
		m_splits[iLftChild + 1] = rgtChild;
		
		setNodeInternal(iNode, plane->getAxis(), plane->getPos());
		clearSplit(iNode, level);
		
		needNextLevel = true;
    }
	return needNextLevel;
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::createLeaf(SahSplit<T> * parent, int level)
{
	if(parent->isEmpty()) {
	
	}
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::clearSplit(int idx, int level)
{
	if(level > 0) {
		delete m_splits[idx];
		m_splits[idx] = NULL;
	}
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::setNodeInternal(int idx, int axis, float pos)
{
	m_node->setInternal(idx, axis, pos, idx + 1);
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::setNodeLeaf(int idx)
{
	m_node->setLeaf(idx);
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
class KdNBuilder {
	int m_branchIdx;
public:
	KdNBuilder();
	virtual ~KdNBuilder();
	
	void build(SahSplit<T> * parent, Tn * nodes);
	void subdivide(Tn * node);
protected:
	void addBranch()
	{ m_branchIdx += 2; }
	
private:

};

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
KdNBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::KdNBuilder()
{
	
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
KdNBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::~KdNBuilder()
{
	
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdNBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::build(SahSplit<T> * parent, Tn * nodes)
{
	KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn> treelet;
	treelet.build(parent, &nodes[0]);
	m_branchIdx = 1;
	subdivide(&nodes[0]);
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdNBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::subdivide(Tn * node)
{	
	node->verbose();
	
}
//:~
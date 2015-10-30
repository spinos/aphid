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
	static int LevelOffset[NumLevels+1];
    
	///
	///           parent
    ///       [0           1]root         level 0
	///       
    ///   [2    3]0      [4      5]1      level 1
	///
    ///  [6 7]2 [8 9]3  [10 11]4 [12 13]5 level 2
    ///
	///  []6[]7 []8[]9  []10[]11 []12[]13 level 3
	///
    SahSplit<T> * m_splits[(1<<NumLevels+1) - 1];
	
public:
	KdTreelet();
	virtual ~KdTreelet();
	
	void build(SahSplit<T> * parent, Tn * node, Tn * root, int iRoot, int offsetRoot);
	static int NumNodes();
	static int LastLevelOffset();
	
	SahSplit<T> * split(int idx);
protected:
	bool subdivideRoot(SahSplit<T> * parent, Tn * root, int iRoot, int offsetRoot);
	bool subdivideInterial(Tn * interial, int level);
	void setNodeInternal(Tn * node, int idx, int axis, float pos, int offset);
	void setNodeLeaf(SahSplit<T> * parent, Tn * node, int idx);
private:
	void clearSplit(int idx);
	void costNotice(SahSplit<T> * parent, SplitEvent * plane) const;
};

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
int KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::NumPrimsInLeaf = 1<<NumPrimsInLeafLog;

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
int KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::NumSubSplits = (1<<NumLevels+1) - 1;

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
int KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::LevelOffset[NumLevels+1];

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::KdTreelet()
{
	int i;
	for(i=0;i<NumSubSplits;i++) {
		m_splits[i] = NULL;
	}
	int a = 0;
	for(i=1;i<=NumLevels;i++) {
		LevelOffset[i] = a;
		a += 1<<i;
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
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::build(SahSplit<T> * parent, Tn * node, Tn * root, int iRoot, int offsetRoot)
{
	if(!subdivideRoot(parent, root, iRoot, offsetRoot)) return;
	
    int level = 1;
    for(;level < NumLevels; level++) {
        if(!subdivideInterial(node, level)) break;
	}
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
bool KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::subdivideRoot(SahSplit<T> * parent, Tn * root, int iRoot, int offsetRoot)
{
	if(parent->numPrims() <= NumPrimsInLeaf) {
		setNodeLeaf(parent, root, iRoot);
		return false;
	}

	SplitEvent * plane = parent->bestSplit();
	
	if(plane->getCost() > parent->visitCost()) {
		costNotice(parent, plane);
		setNodeLeaf(parent, root, iRoot);
		return false;
	}
	
	SahSplit<T>  * lftChild = new SahSplit<T>(plane->leftCount());
	SahSplit<T>  * rgtChild = new SahSplit<T>(plane->rightCount());
	parent->partition(lftChild, rgtChild);
	
	m_splits[0] = lftChild;
	m_splits[1] = rgtChild;
	
	std::cout<<"\n root offset "<<offsetRoot;
	setNodeInternal(root, iRoot, plane->getAxis(), plane->getPos(), offsetRoot | Tn::TreeletOffsetMask);
	
	return true;
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
bool KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::subdivideInterial(Tn * interial, int level)
{
	bool needNextLevel = false;
    std::cout<<"\n\n subdiv level "<<level;
    const int nSplitAtLevel = 1<<level;
    int i;
    for(i=0; i<nSplitAtLevel; i++) {
        const int iNode = LevelOffset[level] + i;
        const int iLftChild = iNode + iNode + 2;
        std::cout<<"\n split node "<<iNode;
        // std::cout<<" child offset "<<iLftChild;
        
        SahSplit<T>  * parent = m_splits[iNode];
		
		if(parent->numPrims() <= NumPrimsInLeaf) {
			setNodeLeaf(parent, interial, iNode);
			clearSplit(iNode);
			continue;
		}
	
        SplitEvent * plane = parent->bestSplit();
		
		if(plane->getCost() > parent->visitCost()) {
			costNotice(parent, plane);
			setNodeLeaf(parent, interial, iNode);
			clearSplit(iNode);
			continue;
		}
		
		if(level < NumLevels) {
			SahSplit<T>  * lftChild = new SahSplit<T>(plane->leftCount());
			SahSplit<T>  * rgtChild = new SahSplit<T>(plane->rightCount());
			parent->partition(lftChild, rgtChild);
			
			m_splits[iLftChild] = lftChild;
			m_splits[iLftChild + 1] = rgtChild;
			
			clearSplit(iNode);
		}
		
		setNodeInternal(interial, iNode, plane->getAxis(), plane->getPos(), iNode + 2);
		
		needNextLevel = true;
    }
	return needNextLevel;
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::clearSplit(int idx)
{
	delete m_splits[idx];
	m_splits[idx] = NULL;
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::setNodeInternal(Tn * node, int idx, int axis, float pos, int offset)
{ node->setInternal(idx, axis, pos, offset); }

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::setNodeLeaf(SahSplit<T> * parent, Tn * node, int idx)
{
	if(!parent->isEmpty()) {
	
	}
	node->setLeaf(idx);
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
int KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::NumNodes()
{ return (1<<NumLevels+1) - 2; }

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
int KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::LastLevelOffset()
{ return LevelOffset[NumLevels];}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
SahSplit<T> * KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::split(int idx)
{ return m_splits[idx]; }

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn>::costNotice(SahSplit<T> * parent, SplitEvent * plane) const
{
	std::cout<<"\n visit cost "
			<<parent->visitCost()
			<<" < split cost "
			<<plane->getCost()
			<<" stop subdivide\n";
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
class KdNBuilder {
	int m_branchIdx;
public:
	KdNBuilder();
	virtual ~KdNBuilder();
	
	void build(SahSplit<T> * parent, Tn * nodes);
	void subdivide(KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn> * treelet, Tn * node);
protected:
	void addBranch()
	{ m_branchIdx += 2; }
	
private:

};

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
KdNBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::KdNBuilder() {}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
KdNBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::~KdNBuilder() {}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdNBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::build(SahSplit<T> * parent, Tn * nodes)
{
	KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn> treelet;
	Tn * root = &nodes[0];
	treelet.build(parent, &nodes[1], root, 0, 1);
	root->verbose();
	m_branchIdx = 1;
	subdivide(&treelet, &nodes[1]);
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdNBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::subdivide(KdTreelet<NumLevels, NumPrimsInLeafLog, T, Tn> * treelet, Tn * node)
{	
	const int n = treelet->NumNodes();
	int i = treelet->LastLevelOffset();
	for(;i<n;i++) {
		SahSplit<T> * parent = treelet->split(i);
		if(parent) parent->verbose();
	}
}
//:~
/*
 *  KdBuilder.h
 *  aphid
 *
 *  Created by jian zhang on 10/29/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "Treelet.h"
#include "KdSah.h"

template<int NumLevels, typename T, typename Tn>
class KdTreeletBuilder : public Treelet<NumLevels > {
	static int NumSubSplits;
    SahSplit<T> * m_splits[(1<<NumLevels+1) - 1];
	int m_index;
    
public:
	KdTreeletBuilder(int index);
	virtual ~KdTreeletBuilder();
	
	void build(int parentIdx, SahSplit<T> * parent, Tn * node, Tn * root, int iRoot);
	
	SahSplit<T> * split(int idx);
    void setIndex(int x);
    int index() const;
    
	static int NumPrimsInLeaf;
    static int TotalNumLeafPrims;
	
protected:
	bool subdivideRoot(int parentIdx, SahSplit<T> * parent, Tn * root, int iRoot);
	bool subdivideInterial(Tn * interial, int level);
	void setNodeInternal(Tn * node, int idx, int axis, float pos, int offset);
	void setNodeLeaf(SahSplit<T> * parent, Tn * node, int idx);
private:
	void clearSplit(int idx);
	void costNotice(SahSplit<T> * parent, SplitEvent * plane) const;
};

template<int NumLevels, typename T, typename Tn>
int KdTreeletBuilder<NumLevels, T, Tn>::NumPrimsInLeaf = 8;

template<int NumLevels, typename T, typename Tn>
int KdTreeletBuilder<NumLevels, T, Tn>::NumSubSplits = (1<<NumLevels+1) - 1;

template<int NumLevels, typename T, typename Tn>
int KdTreeletBuilder<NumLevels, T, Tn>::TotalNumLeafPrims = 0;

template<int NumLevels, typename T, typename Tn>
KdTreeletBuilder<NumLevels, T, Tn>::KdTreeletBuilder(int index)
{
	int i;
	for(i=0;i<NumSubSplits;i++) {
		m_splits[i] = NULL;
	}
    m_index = index;
}

template<int NumLevels, typename T, typename Tn>
KdTreeletBuilder<NumLevels, T, Tn>::~KdTreeletBuilder()
{
	int i;
	for(i=1;i<NumSubSplits;i++) {
		if(m_splits[i]) delete m_splits[i];
	}
}

template<int NumLevels, typename T, typename Tn>
void KdTreeletBuilder<NumLevels, T, Tn>::build(int parentIdx, SahSplit<T> * parent, Tn * node, Tn * root, int iRoot)
{
	if(!subdivideRoot(parentIdx, parent, root, iRoot)) return;
	
    int level = 1;
    for(;level <= NumLevels; level++) {
        if(!subdivideInterial(node, level)) break;
	}
}

template<int NumLevels, typename T, typename Tn>
bool KdTreeletBuilder<NumLevels, T, Tn>::subdivideRoot(int parentIdx, SahSplit<T> * parent, Tn * root, int iRoot)
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
	
    const int offsetRoot = index() - parentIdx;
	// std::cout<<"\n root offset "<<offsetRoot;
	setNodeInternal(root, iRoot, plane->getAxis(), plane->getPos(), offsetRoot | Tn::TreeletOffsetMask);
	
	return true;
}

template<int NumLevels, typename T, typename Tn>
bool KdTreeletBuilder<NumLevels, T, Tn>::subdivideInterial(Tn * interial, int level)
{
	bool needNextLevel = false;
    // std::cout<<"\n\n subdiv level "<<level;
    const int nSplitAtLevel = 1<<level;
    int i;
    for(i=0; i<nSplitAtLevel; i++) {
        const int iNode = Treelet<NumLevels>::OffsetByLevel(level) + i;
        const int iLftChild = iNode + Treelet<NumLevels>::ChildOffset(iNode);
        
		// std::cout<<"\n  node "<<iNode;
		
        SahSplit<T>  * parent = m_splits[iNode];
		if(!parent) {
			// std::cout<<"\n no parent ";
			continue;
		}
        
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
			
			// std::cout<<"\n spawn "
			//	<<" lft "<<iLftChild
			//	<<" rgt "<<iLftChild + 1;
			
			clearSplit(iNode);
			setNodeInternal(interial, iNode, plane->getAxis(), plane->getPos(), Treelet<NumLevels>::ChildOffset(iNode) );
		}
		else {
			// std::cout<<"\n end of treelet "<<parent->numPrims();
			setNodeInternal(interial, iNode, 0, 0.f, 1);
		}

		needNextLevel = true;
    }
	return needNextLevel;
}

template<int NumLevels, typename T, typename Tn>
void KdTreeletBuilder<NumLevels, T, Tn>::clearSplit(int idx)
{
	delete m_splits[idx];
	m_splits[idx] = NULL;
}

template<int NumLevels, typename T, typename Tn>
void KdTreeletBuilder<NumLevels, T, Tn>::setNodeInternal(Tn * node, int idx, int axis, float pos, int offset)
{ node->setInternal(idx, axis, pos, offset); }

template<int NumLevels, typename T, typename Tn>
void KdTreeletBuilder<NumLevels, T, Tn>::setNodeLeaf(SahSplit<T> * parent, Tn * node, int idx)
{
	if(!parent->isEmpty()) {
		// std::cout<<"\n leaf n prims "<<parent->numPrims();
	}
	node->setLeaf(idx);
	TotalNumLeafPrims += parent->numPrims();
}

template<int NumLevels, typename T, typename Tn>
SahSplit<T> * KdTreeletBuilder<NumLevels, T, Tn>::split(int idx)
{ return m_splits[idx]; }

template<int NumLevels, typename T, typename Tn>
void KdTreeletBuilder<NumLevels, T, Tn>::costNotice(SahSplit<T> * parent, SplitEvent * plane) const
{
	std::cout<<"\n visit cost "
			<<parent->visitCost()
			<<" < split cost "
			<<plane->getCost()
			<<" stop subdivide\n";
}

template<int NumLevels, typename T, typename Tn>
void KdTreeletBuilder<NumLevels, T, Tn>::setIndex(int x)
{ m_index = x; }

template<int NumLevels, typename T, typename Tn>
int KdTreeletBuilder<NumLevels, T, Tn>::index() const
{ return m_index; }



template<int NumLevels, typename T, typename Tn>
class KdNBuilder {
	int m_branchIdx;
public:
	KdNBuilder();
	virtual ~KdNBuilder();
	
	void build(SahSplit<T> * parent, Tn * nodes);
	void subdivide(KdTreeletBuilder<NumLevels, T, Tn> * treelet, Tn * nodes);
	
	static void SetNumPrimsInLeaf(int x);
	
protected:
	void addBranch()
	{ m_branchIdx++; }
	
private:

};

template<int NumLevels, typename T, typename Tn>
KdNBuilder<NumLevels, T, Tn>::KdNBuilder() {}

template<int NumLevels, typename T, typename Tn>
KdNBuilder<NumLevels, T, Tn>::~KdNBuilder() {}

template<int NumLevels, typename T, typename Tn>
void KdNBuilder<NumLevels, T, Tn>::SetNumPrimsInLeaf(int x)
{ KdTreeletBuilder<NumLevels, T, Tn>::NumPrimsInLeaf = x; }

template<int NumLevels, typename T, typename Tn>
void KdNBuilder<NumLevels, T, Tn>::build(SahSplit<T> * parent, Tn * nodes)
{
	m_branchIdx = 1;
	KdTreeletBuilder<NumLevels, T, Tn>::TotalNumLeafPrims = 0;
    KdTreeletBuilder<NumLevels, T, Tn> treelet(1);
	Tn * root = &nodes[0];
	/// only first node in first treelet is useful
	/// spawn into second treelet
	treelet.build(0, parent, &nodes[1], root, 0);
	subdivide(&treelet, nodes);
    std::cout<<"\n n tree nodes "<<m_branchIdx+1;
	std::cout<<"\n n leaf prims "<<KdTreeletBuilder<NumLevels, T, Tn>::TotalNumLeafPrims;
}

template<int NumLevels, typename T, typename Tn>
void KdNBuilder<NumLevels, T, Tn>::subdivide(KdTreeletBuilder<NumLevels, T, Tn> * treelet, Tn * nodes)
{	
    const int parentIdx = treelet->index();
    Tn * parentNode = &nodes[parentIdx];
	const int n = treelet->numNodes();
	int i = treelet->LastLevelOffset();
	for(;i<n;i++) {
		SahSplit<T> * parent = treelet->split(i);
		if(!parent) continue;
		
		addBranch();

        KdTreeletBuilder<NumLevels, T, Tn> subTreelet(m_branchIdx);
        subTreelet.build(parentIdx, parent, &nodes[m_branchIdx], parentNode, i);
		subdivide(&subTreelet, nodes);
	}
}
//:~
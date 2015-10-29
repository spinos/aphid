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
class KdBuilder {
	static int NumPrimsInLeaf;
    static int NumSubSplits;
    SahSplit<T> * m_splits[(1<<NumLevels) - 1];
public:
	KdBuilder() {}
	virtual ~KdBuilder() {}
	
	void build(SahSplit<T> * parent);
protected:
	void subdivide(SahSplit<T> * parent, int level);
    void subdivide1(int level);
private:
    ///
    ///       [1            2]0
    ///   [3    4]1      [5      6]2
    ///  [7 8]3 [9 10]4  [11 12]5 [13 14]6
    ///
    static int NodeOffset(int level, int idx)
    {
        int x = 0;
        int i = 0;
        while(i<level) {
            x += 1<<i;
            i++;
        }
        return x + idx;
    }
    
    static int ChildOffset(int level, int idx)
    { return NodeOffset(level, idx) + 1; }
    
};

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
int KdBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::NumPrimsInLeaf = 1<<NumPrimsInLeafLog;

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
int KdBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::NumSubSplits = (1<<NumLevels) - 1;

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::build(SahSplit<T> * parent)
{
    m_splits[0] = parent;
    int level = 0;
    for(;level < NumLevels; level++)
        subdivide1(level);
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::subdivide1(int level)
{
    std::cout<<"\n subdiv level "<<level;
    const int nSplitAtLevel = 1<<level;
    int i;
    for(i=0; i<nSplitAtLevel; i++) {
        std::cout<<"\n split node "<<NodeOffset(level, i);
        std::cout<<" child offset "<<ChildOffset(level, i);
        const int iNode = NodeOffset(level, i);
        const int iLftChild = iNode + ChildOffset(level, i);
        
        SahSplit<T>  * parent = m_splits[iNode];
        SplitEvent * plane = parent->bestSplit();
        SahSplit<T>  * lftChild = new SahSplit<T>(plane->leftCount());
        SahSplit<T>  * rgtChild = new SahSplit<T>(plane->rightCount());
        parent->partition(lftChild, rgtChild);
    
        if(level < NumLevels-1) {
            m_splits[iLftChild] = lftChild;
            m_splits[iLftChild + 1] = rgtChild;
        }
        else {
            
        }
    }
}

template<int NumLevels, int NumPrimsInLeafLog, typename T, typename Tn>
void KdBuilder<NumLevels, NumPrimsInLeafLog, T, Tn>::subdivide(SahSplit<T> * parent, int level)
{
	if(parent->numPrims() <= NumPrimsInLeaf || level == 9) return;
	
	SplitEvent * plane = parent->bestSplit();
	
	if(plane->getCost() > parent->visitCost()) {
		std::cout<<"\n visit cost "
				<<parent->visitCost()
				<<" < split cost "
				<<plane->getCost()
				<<" stop subdivide\n";
		return;
	}
	
	std::cout<<"\n level "
			<<level
			<<" split "<<parent->numPrims();
    // plane->verbose();
	SahSplit<T>  * lftChild = new SahSplit<T>(plane->leftCount());
	SahSplit<T>  * rgtChild = new SahSplit<T>(plane->rightCount());
	parent->partition(lftChild, rgtChild);
	
	if(plane->leftCount() > 0) {
		subdivide(lftChild, level+1);
	}
	else {
	
	}
	
	delete lftChild;
	
	if(plane->rightCount() > 0) {
		subdivide(rgtChild, level+1);
	}
	else {
	
	}
	
	delete rgtChild;
}
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

template<int NumPrimsInLeafLog, typename T>
class KdBuilder {
	static int NumPrimsInLeaf;
public:
	KdBuilder() {}
	virtual ~KdBuilder() {}
	
	void build(SahSplit<T> * parent);
protected:
	void subdivide(SahSplit<T> * parent, int level);
private:

};

template<int NumPrimsInLeafLog, typename T>
int KdBuilder<NumPrimsInLeafLog, T>::NumPrimsInLeaf = 1<<NumPrimsInLeafLog;

template<int NumPrimsInLeafLog, typename T>
void KdBuilder<NumPrimsInLeafLog, T>::build(SahSplit<T> * parent)
{
	int level = 0;
	subdivide(parent, level);
}

template<int NumPrimsInLeafLog, typename T>
void KdBuilder<NumPrimsInLeafLog, T>::subdivide(SahSplit<T> * parent, int level)
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
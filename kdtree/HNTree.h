/*
 *  HNTree.h
 *  julia
 *
 *  Created by jian zhang on 3/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "KdNTree.h"
#include <HBase.h>
#include <HOocArray.h>

namespace aphid {

class HBaseNTree : public HBase {

public:
	HBaseNTree(const std::string & name);
	virtual ~HBaseNTree();
	
	virtual char verifyType();

protected:
};

template <typename T, typename Tn>
class HNTree : public KdNTree<T, Tn>, public HBaseNTree {
	
public:
	HNTree(const std::string & name);
	virtual ~HNTree();
	
	virtual char save();
	virtual char load();
    
protected:
	void save240Node();
	void saveLeaf();
	void saveInd();
	void saveRope();
	
	void load240Node();
	void loadLeaf();
	void loadInd();
	void loadRope();
	
private:

};

template <typename T, typename Tn>
HNTree<T, Tn>::HNTree(const std::string & name) :
HBaseNTree(name)
{}

template <typename T, typename Tn>
HNTree<T, Tn>::~HNTree() {}

template <typename T, typename Tn>
char HNTree<T, Tn>::save()
{
	if(sizeof(Tn) == 240) 
		save240Node();
	
	saveLeaf();
	saveInd();
	saveRope();
	
	if(!hasNamedAttr(".bbx") )
	    addFloatAttr(".bbx", 6);
	writeFloatAttr(".bbx", (float *)&KdNTree<T, Tn>::getBBox() );
	
	return 1;
}

template <typename T, typename Tn>
void HNTree<T, Tn>::save240Node()
{
	HOocArray<hdata::TChar, 240, 256> treeletD(".node");
	if(hasNamedData(".node") )
		treeletD.openStorage(fObjectId, true);
	else
		treeletD.createStorage(fObjectId);
	
	const sdb::VectorArray<Tn> & src = KdNTree<T, Tn>::nodes();
	int n = KdNTree<T, Tn>::numBranches();
	int i=0;
	for(;i<n;++i) {
		treeletD.insert((char *)src[i] );
	}
	
	treeletD.finishInsert();
	
	if(!hasNamedAttr(".nnode") )
	    addIntAttr(".nnode", 1);
	writeIntAttr(".nnode", &n);
	std::cout<<"\n save "<<n<<" node240";
}
	
template <typename T, typename Tn>
void HNTree<T, Tn>::saveLeaf()
{
	HOocArray<hdata::TInt, 8, 256> leafD(".leaf");
	if(hasNamedData(".leaf") )
		leafD.openStorage(fObjectId, true);
	else
		leafD.createStorage(fObjectId);
	
	const sdb::VectorArray<knt::TreeLeaf> & src = KdNTree<T, Tn>::leafNodes();
	int n = KdNTree<T, Tn>::numLeafNodes();
	int i=0;
	for(;i<n;++i) {
		leafD.insert((char *)src[i] );
	}
	
	leafD.finishInsert();
	
	if(!hasNamedAttr(".nleaf") )
	    addIntAttr(".nleaf", 1);
	writeIntAttr(".nleaf", &n);
	std::cout<<"\n save "<<n<<" leaf";
}

template <typename T, typename Tn>
void HNTree<T, Tn>::saveInd()
{
	HOocArray<hdata::TInt, 64, 64> indD(".ind");
	if(hasNamedData(".ind") )
		indD.openStorage(fObjectId, true);
	else
		indD.createStorage(fObjectId);
	
	const sdb::VectorArray<int> & src = KdNTree<T, Tn>::primIndirection();
	int n = KdNTree<T, Tn>::numPrimIndirection();
	
	int b[64];
	int i=0, j=0;
	for(;i<n;++i) {
		b[j++] = *src[i];
		if(j==64) {
			indD.insert((char *)b );
			j=0;
		}
	}
	
	if(j>0) indD.insert((char *)b );
	
	indD.finishInsert();
	
	if(!hasNamedAttr(".nind") )
	    addIntAttr(".nind", 1);
	writeIntAttr(".nind", &n);
	std::cout<<"\n save "<<n<<" indirection";
}

template <typename T, typename Tn>
void HNTree<T, Tn>::saveRope()
{
	HOocArray<hdata::TFloat, 8, 256> ropeD(".rope");
	if(hasNamedData(".rope") )
		ropeD.openStorage(fObjectId, true);
	else
		ropeD.createStorage(fObjectId);
	
	const BoundingBox * src = KdNTree<T, Tn>::ropes();
	int n = KdNTree<T, Tn>::numRopes();
	int i=0;
	for(;i<n;++i) {
		ropeD.insert((char *)&src[i] );
	}
	
	ropeD.finishInsert();
	
	if(!hasNamedAttr(".nrope") )
	    addIntAttr(".nrope", 1);
	writeIntAttr(".nrope", &n);
	std::cout<<"\n save "<<n<<" rope";
}

template <typename T, typename Tn>
char HNTree<T, Tn>::load()
{
	BoundingBox b;
	readFloatAttr(".bbx", (float *)&b );
	KdNTree<T, Tn>::clear(b);
	std::cout<<"\n bbox "<<b;
	
	if(sizeof(Tn) == 240) 
		load240Node();
		
	loadLeaf();
	loadInd();
	loadRope();
	
	return 1;
}

template <typename T, typename Tn>
void HNTree<T, Tn>::load240Node()
{
	int n =0;
	readIntAttr(".nnode", &n);
	
	HOocArray<hdata::TChar, 240, 256> treeletD(".node");
	if(!treeletD.openStorage(fObjectId)) 
		return;
	
	int i=0;
	for(;i<n;++i) {
		treeletD.readColumn((char *)KdNTree<T, Tn>::addTreelet(), i);
	}
	
	std::cout<<"\n load "<<n<<" node240";
}

template <typename T, typename Tn>
void HNTree<T, Tn>::loadLeaf()
{
	int n =0;
	readIntAttr(".nleaf", &n);
	
	HOocArray<hdata::TInt, 8, 256> leafD(".leaf");
	if(!leafD.openStorage(fObjectId))
		return;
	
	int i=0;
	for(;i<n;++i) {
		leafD.readColumn((char *)KdNTree<T, Tn>::addLeaf(), i);
	}
	
	std::cout<<"\n load "<<n<<" leaf";
}

template <typename T, typename Tn>
void HNTree<T, Tn>::loadInd()
{
	int n =0;
	readIntAttr(".nind", &n);
	
	HOocArray<hdata::TInt, 64, 64> indD(".ind");
	if(!indD.openStorage(fObjectId))
		return;
	
	int b[64];
	int i=0, j=0;
	for(;i<n;++i) {
		if((i & 63) == 0) {
			indD.readColumn((char *)b, j);
			j++;
		}
		*KdNTree<T, Tn>::addIndirection() = b[i & 63];
	}
	
	std::cout<<"\n load "<<n<<" indirection";
}
	
template <typename T, typename Tn>
void HNTree<T, Tn>::loadRope()
{
	int n =0;
	readIntAttr(".nrope", &n);
	
	HOocArray<hdata::TFloat, 8, 256> ropeD(".rope");
	if(!ropeD.openStorage(fObjectId))
		return;
	
	KdNTree<T, Tn>::createRopes(n);
	int i=0;
	for(;i<n;++i) {
		ropeD.readColumn((char *)KdNTree<T, Tn>::ropesR(i), i);
	}
	
	std::cout<<"\n load "<<n<<" rope";
}

}
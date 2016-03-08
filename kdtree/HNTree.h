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

template <typename T, typename Tn>
class HNTree : public KdNTree<T, Tn>, public HBase {
	
public:
	HNTree(const std::string & name);
	virtual ~HNTree();
	
	virtual char save();
	//virtual char load();
    //virtual char verifyType();

protected:
	void save240Node();
	void saveLeaf();
	void saveInd();
	void saveRope();
	
private:

};

template <typename T, typename Tn>
HNTree<T, Tn>::HNTree(const std::string & name) :
HBase(name)
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
	if(hasNamedData(".node") ) {
		treeletD.openStorage(fObjectId);
		treeletD.clear();
	}
	else {
		treeletD.createStorage(fObjectId);
	}
	
	const sdb::VectorArray<Tn> & src = KdNTree<T, Tn>::nodes();
	int n = KdNTree<T, Tn>::numNodes();
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
	if(hasNamedData(".leaf") ) {
		leafD.openStorage(fObjectId);
		leafD.clear();
	}
	else {
		leafD.createStorage(fObjectId);
	}
	
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
	if(hasNamedData(".ind") ) {
		indD.openStorage(fObjectId);
		indD.clear();
	}
	else {
		indD.createStorage(fObjectId);
	}
	
	const sdb::VectorArray<int> & src = KdNTree<T, Tn>::leafIndirection();
	int n = KdNTree<T, Tn>::numLeafIndirection();
	
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
	if(hasNamedData(".rope") ) {
		ropeD.openStorage(fObjectId);
		ropeD.clear();
	}
	else {
		ropeD.createStorage(fObjectId);
	}
	
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

}
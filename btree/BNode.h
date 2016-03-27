/*
 *  BNode.h
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <Entity.h>
#include <Types.h>
#include <sstream>

namespace aphid {

namespace sdb {

#define DBG_SPLIT 1

class TreeNode : public Entity
{
/// to left child or
/// to right sibling for leaf
	Entity * m_link;
	bool m_isLeaf;
	
public:
	TreeNode(Entity * parent = NULL);
	virtual ~TreeNode();
	
	bool isRoot() const;
	bool hasChildren() const;
	bool isLeaf() const;
	
	Entity * sibling() const;
	Entity * leftChild() const;
	
	void setLeaf();
	void connectSibling(Entity * another);
	void connectLeftChild(Entity * another);
	
	static int MaxNumKeysPerNode;
	static int MinNumKeysPerNode;
	
protected:
	struct NodeIndirection {
		void reset() {
			_p = NULL;
		}
		
		void take(Entity * src) {
			_p = src;
		}
		
		Entity * give() {
			return _p;
		}
		
		Entity * _p;
	};
	
private:
	
};

class SearchResult
{
public:
	int found, low, high;
};

template <typename KeyType>
class BNode : public TreeNode
{
	Pair<KeyType, Entity> * m_data;
	int m_numKeys;
	
public:
	BNode(Entity * parent = NULL);
	virtual ~BNode();
	
	Pair<KeyType, Entity> * insert(const KeyType & x);
    Pair<Entity *, Entity> find(const KeyType & x);
	void remove(const KeyType & x);
	
    void getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const;
	BNode * firstLeaf();
	BNode * nextLeaf();
	
	Pair<Entity *, Entity> findInNode(const KeyType & x, SearchResult * result);
	
	const int numKeys() const  { return m_numKeys; }
	const KeyType & key(const int & i) const { return m_data[i].key; }
	Entity * index(const int & i) const { return m_data[i].index; }
	
	friend std::ostream& operator<<(std::ostream &output, const BNode & p) {
        output << p.str();
        return output;
    }
	
	void dbgFind(const KeyType & x);
	bool dbgLinks(bool silient=true);
	void dbgDown() const;
	
private:	
	const KeyType firstKey() const;
	const KeyType lastKey() const;
	bool isKeyInRange(const KeyType & x) const;
	int keyRight(const KeyType & x) const;
	int keyLeft(const KeyType & x) const;
	
	Pair<KeyType, Entity> firstData() const;
	Pair<KeyType, Entity> lastData() const;
	const Pair<KeyType, Entity> data(int x) const;
	Pair<KeyType, Entity> * dataP(int x) const;
	const Pair<KeyType, Entity> dataRightTo(const KeyType & x) const;
	bool dataLeftTo(const KeyType & x, Pair<KeyType, Entity> & dst) const;
	
	void setData(int k, const Pair<KeyType, Entity> & x);
	void insertData(Pair<KeyType, Entity> x);
	
	BNode * parentNode() const;
	BNode * siblingNode() const;
	BNode * ancestor(const KeyType & x, bool & found) const;
	BNode * leftTo(const KeyType & x) const;
	BNode * rightTo(const KeyType & x, Pair<KeyType, Entity> & k) const;
	BNode * leafLeftTo(const KeyType & x);
    BNode * nextIndex(KeyType x) const;
	BNode * leftInteriorNeighbor() const;
	BNode * rightInteriorNeighbor(int & indParent) const;
	BNode * linkedNode(int idx) const;
	BNode * leftChildNode() const; 
	
	Pair<KeyType, Entity> * insertRoot(const KeyType & x);
	Pair<KeyType, Entity> * insertLeaf(const KeyType & x);
	Pair<KeyType, Entity> * insertInterior(const KeyType & x);
	
	void splitRootToLeaves();
	BNode *splitLeaf(const KeyType & x);
	
	void insertKey(KeyType x);
	void removeFirstData1();
	bool removeKey(const KeyType & x);
	bool removeKeyAndData(const KeyType & x);

	void partRoot(Pair<KeyType, Entity> x);
	Pair<KeyType, Entity> partData(Pair<KeyType, Entity> x, Pair<KeyType, Entity> old[], BNode * lft, BNode * rgt, bool doSplitLeaf = false);
	
	void partInterior(Pair<KeyType, Entity> x);
	
/// for each child, set parent
	void connectToChildren();
/// insert data, split if needed
	void bounce(const Pair<KeyType, Entity> b);

	bool leafBalance();
	bool leafBalanceRight();
	bool leafBalanceLeft();
	void sendDataRight(int num, BNode * rgt);
	void sendDataLeft(int num, BNode * lft);
	
	void removeLastData();
	void removeFirstData();

	void replaceKey(KeyType x, KeyType y);
	bool hasKey(const KeyType & x) const;
	
	void removeRoot(const KeyType & x);
	void removeLeaf(const KeyType & x);
	bool removeDataLeaf(const KeyType & x);

	bool mergeLeaf();
	bool mergeLeafRight();
	bool mergeLeafLeft();
	
	void pop(const Pair<KeyType, Entity> & x);
	void popRoot(const Pair<KeyType, Entity> & x);
	void popInterior(const Pair<KeyType, Entity> & x);
	
	void mergeData(BNode * another, int start = 0);
	void replaceIndex(int n, Pair<KeyType, Entity> x);
	
	bool interiorBalance();
	
	bool mergeInterior(const KeyType x);
	bool mergeInteriorRight(const KeyType x);
	bool mergeInteriorLeft(const KeyType x);
	
	bool interiorBalanceRight();
	bool interiorBalanceLeft();
	
	bool isFull() const;
	bool underflow() const;
	void reduceNumKeys();
	void increaseNumKeys();
	void setNumKeys(int x);
	bool shouldMerge(BNode * lft, BNode * rgt) const;
	bool shouldInteriorMerge(BNode * lft, BNode * rgt) const;
	int shouldBalance(BNode * lft, BNode * rgt) const;
	
	const SearchResult findKey(const KeyType & x) const;
	
	Pair<Entity *, Entity> findRoot(const KeyType & x);
	Pair<Entity *, Entity> findLeaf(const KeyType & x);
	Pair<Entity *, Entity> findInterior(const KeyType & x);
	
	void dbgFindInRoot(const KeyType & x);
	void dbgFindInLeaf(const KeyType & x);
	void dbgFindInInterior(const KeyType & x);
		
	const std::string str() const;
	void deepPrint() const;
   
	bool checkDupKey(KeyType & duk) const;
	void dbgUp();
	void dbgRightUp(BNode<KeyType> * rgt);
	int countBalance(int a, int b);
	void validateKeys();
	void behead();
	
};

template <typename KeyType>  
BNode<KeyType>::BNode(Entity * parent) : TreeNode(parent),
m_numKeys(0)
{
	m_data = new Pair<KeyType, Entity>[MaxNumKeysPerNode];
	for(int i=0;i< MaxNumKeysPerNode;++i)
        m_data[i].index = NULL;
}

template <typename KeyType> 
BNode<KeyType>::~BNode()
{
	for(int i=0;i< numKeys();++i) {
		if(m_data[i].index) delete m_data[i].index;
	}
	delete[] m_data;
}

template <typename KeyType>  
BNode<KeyType> * BNode<KeyType>::nextIndex(KeyType x) const
{
	//std::cout<<"\n find "<<x<<" in "<<*this;
	
	if(firstKey() > x) {
		if(!leftChild()) std::cout<<"\n error first index";
		return static_cast<BNode *>(leftChild());
	}
	int ii;
	SearchResult s = findKey(x);
	ii = s.low;
	if(s.found > -1) ii = s.found;
	else if(key(s.high) < x) ii = s.high;
	
	//std::cout<<"found "<<ii<<"\n";
	return static_cast<BNode *>(m_data[ii].index);
}

template <typename KeyType> 
const Pair<KeyType, Entity> BNode<KeyType>::data(int x) const 
{ return m_data[x]; }

template <typename KeyType> 
Pair<KeyType, Entity> * BNode<KeyType>::dataP(int x) const 
{ return &m_data[x]; }

template <typename KeyType> 
const KeyType BNode<KeyType>::firstKey() const 
{ return m_data[0].key; }

template <typename KeyType> 
const KeyType BNode<KeyType>::lastKey() const 
{ return m_data[numKeys() - 1].key; }

template <typename KeyType> 
Pair<KeyType, Entity> * BNode<KeyType>::insert(const KeyType & x)
{
	if(isRoot()) 
		return insertRoot(x);
	
	return insertInterior(x);
}

template <typename KeyType> 
void BNode<KeyType>::remove(const KeyType & x)
{
	if(isRoot()) {
    
		removeRoot(x);
	}
	else if(isLeaf()) {
    
		removeLeaf(x);
	}
	else {
    
		BNode * n = nextIndex(x);
		n->remove(x);
	}
}

template <typename KeyType> 
void BNode<KeyType>::removeRoot(const KeyType & x)
{
	if(hasChildren()) {
		BNode * n = nextIndex(x);
		n->remove(x);
	}
	else {
		removeKeyAndData(x);
		
		connectLeftChild(NULL);
	}
}

template <typename KeyType> 
Pair<KeyType, Entity> * BNode<KeyType>::insertRoot(const KeyType & x)
{
	if(hasChildren()) {
		BNode * n = nextIndex(x);
		if(n->isLeaf() ) {
			return n->insertLeaf(x);
		}
		return n->insertInterior(x);
	}

/// a single node
	if(isFull()) {	
		splitRootToLeaves();
		BNode * n = nextIndex(x);
		return n->insertLeaf(x);
	}
	
	if(!hasKey(x)) {
		insertKey(x);
		// std::cout<<"\n bnode insert root "<<x<<" into "<<*this;
	}
	
	return dataP(findKey(x).found);
}

template <typename KeyType> 
Pair<KeyType, Entity> * BNode<KeyType>::insertLeaf(const KeyType & x)
{
	if(!hasKey(x)) {
		insertKey(x);
		/// std::cout<<"\n bnode insert leaf"<<x<<" into "<<*this;
	}
	
	BNode * dst = this;
	if(isFull())
		dst = splitLeaf(x);
	
	
	return dst->dataP(dst->findKey(x).found);
}

template <typename KeyType> 
Pair<KeyType, Entity> * BNode<KeyType>::insertInterior(const KeyType & x)
{
	//std::cout<<"\n insert inner"<<x<<" into "<<*this;
	BNode * n = nextIndex(x);
	if(n->isLeaf() ) {
		//std::cout<<"\n child is leaf"<<n<<" nk "<<n->numKeys();
		return n->insertLeaf(x);
	}
	return n->insertInterior(x);
}

template <typename KeyType> 
void BNode<KeyType>::insertData(Pair<KeyType, Entity> x)
{	
	int i;
    for(i= numKeys() - 1;i >= 0 && m_data[i].key > x.key; i--)
        m_data[i+1] = m_data[i];
		
    m_data[i+1] = x;
	//std::cout<<"insert key "<<x.key<<" at "<<i+1<<"\n";
    increaseNumKeys();
}

template <typename KeyType> 
void BNode<KeyType>::insertKey(KeyType x)
{  
	int i;
    for(i= numKeys() - 1;i >= 0 && m_data[i].key > x; --i) {
		m_data[i+1] = m_data[i];
	}
		
	m_data[i+1].key = x;
	m_data[i+1].index = NULL;
	
	increaseNumKeys();
	
#if 0
	KeyType duk;
	if(!checkDupKey(duk)) {
		std::cout<<" aft insert key "<<x;
		if(duk == x) std::cout<<"\n dupleaf "<<*this;
	}
#endif
}

template <typename KeyType> 
void BNode<KeyType>::splitRootToLeaves()
{
	// std::cout<<"\n split root";
	BNode * one = new BNode(this);
	one->setLeaf();
	BNode * two = new BNode(this); 
	two->setLeaf();
	
	BNode * dst = one;
	int i = 0;
	for(;i < MaxNumKeysPerNode; ++i) {
		
		if(i == MaxNumKeysPerNode / 2)	
			dst = two;
		
		dst->insertData(m_data[i]);
		m_data[i].index = NULL;
	}
	
	/// std::cout<<"\n into "<<*one<<"\n"<<*two;
	
	connectLeftChild(one);
	setNumKeys(1);
	
	m_data[0].key = two->firstKey();
	m_data[0].index = two;
	
	/// std::cout<<"\n aft "<<*this;
	
	one->connectSibling(two);
}

/// 2nd half to new rgt and connect it to old rgt
template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::splitLeaf(const KeyType & x)
{
#if DBG_SPLIT
	//std::cout<<"\n split "<<str();
#endif
	Entity * oldRgt = sibling();
	BNode * two = new BNode(parent()); 
	two->setLeaf();

	for(int i=MaxNumKeysPerNode / 2; i < MaxNumKeysPerNode; ++i) {
		two->insertData(m_data[i]);
		m_data[i].index = NULL;
	}
		
	setNumKeys(MaxNumKeysPerNode / 2);
	
	connectSibling(two);
	if(oldRgt) two->connectSibling(oldRgt);
#if DBG_SPLIT	
	//std::cout<<"\n into "<<str()<<" and "<<two->str();
#endif	
	Pair<KeyType, Entity> b;
	b.key = two->firstKey();
	b.index = two;
	parentNode()->bounce(b);
	
	if(two->hasKey(x)) return two;
	return this;
}

template <typename KeyType> 
void BNode<KeyType>::bounce(const Pair<KeyType, Entity> b)
{	
	if(isFull()) {
		if(isRoot()) 
			partRoot(b);
		else
			partInterior(b);
	}
	else
		insertData(b);
}

template <typename KeyType> 
void BNode<KeyType>::getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const
{
	if(isLeaf()) return;
	if(!hasChildren()) return;
	dst[level].push_back(leftChild());
	for(int i = 0;i < numKeys(); i++) {
		dst[level].push_back(index(i));
    }
		
	level++;
		
	BNode * n = static_cast<BNode *>(leftChild());
	n->getChildren(dst, level);
	for(int i = 0;i < numKeys(); i++) {
		n = static_cast<BNode *>(index(i));
		n->getChildren(dst, level);
	}
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::firstLeaf()
{
	if(isRoot()) { 
		if(hasChildren())
			return static_cast<BNode *>(leftChild())->firstLeaf();
		else 
			return this;
			
	}
	
	if(isLeaf())
		return this;
	
	return static_cast<BNode *>(leftChild())->firstLeaf();
}

template <typename KeyType>
BNode<KeyType> * BNode<KeyType>::nextLeaf() 
{ return static_cast<BNode *>(sibling()); }

template <typename KeyType> 
void BNode<KeyType>::connectToChildren()
{
	if(!hasChildren()) return;
	Entity * n = leftChild();
	n->setParent(this);
	for(int i = 0;i < numKeys(); i++) {
		n = index(i);
		n->setParent(this);
	}
}

/// part into 2 interials
template <typename KeyType> 
void BNode<KeyType>::partRoot(Pair<KeyType, Entity> b)
{
#if DBG_SPLIT
	//std::cout<<"\n part root "<<str()<<"\n add index "<<static_cast<BNode *>(b.index)->firstKey();
#endif

#if DBG_SPLIT
	//std::cout<<"\n right to "<<static_cast<BNode *>(m_data[numKeys() - 1].index)->firstKey();
#endif	

	BNode * one = new BNode(this);
	BNode * two = new BNode(this);
	
	BNode * dst = one;
	const int midI = MaxNumKeysPerNode / 2;
	int i=0;
	for(; i < MaxNumKeysPerNode; ++i) {
		if(i == midI)	
			dst = two;
		else 
			dst->insertData(m_data[i]);
	}
	
	one->connectLeftChild(leftChild());
	
	if(b.key > m_data[midI].key) {
#if DBG_SPLIT
		std::cout<<"\n b.k "<<b.key<<" > 1st.k "<<m_data[midI].key;
#endif
		two->insertData(b);
		
	}
	else {
#if DBG_SPLIT
		std::cout<<"\n b.k "<<b.key<<" < 1st.k "<<m_data[midI].key;
#endif
		one->insertData(b);
	}
	
	two->connectLeftChild(m_data[midI].index);

#if DBG_SPLIT
	//std::cout<<"\n into "<<*one<<" and "<<*two;
	std::cout<<"\n two n k "<<two->numKeys();
	std::cout<<" <- "<<static_cast<BNode *>(two->leftChild() )->firstKey();
	for(int i=0;i<two->numKeys(); ++i) {
		std::cout<<" -> "<<static_cast<BNode *>(two->index(i) )->firstKey();
	}
#endif	
	
	connectLeftChild(one);
	m_data[0].key = m_data[midI].key;
	m_data[0].index = two;

	setNumKeys(1);
#if DBG_SPLIT
	//std::cout<<"\n after "<<*this;
#endif
	one->connectToChildren();
#if DBG_SPLIT	
	//std::cout<<"\n connect right child";
#endif
	two->connectToChildren();
}

template <typename KeyType> 
void BNode<KeyType>::partInterior(Pair<KeyType, Entity> x)
{
#if DBG_SPLIT
	//std::cout<<"part interior "<<*this;
#endif
	BNode * rgt = new BNode(parent());
	
	Pair<KeyType, Entity> * old = new Pair<KeyType, Entity>[MaxNumKeysPerNode];
	for(int i=0; i < MaxNumKeysPerNode; i++)
		old[i] = m_data[i];
	
	setNumKeys(0);
	Pair<KeyType, Entity> p = partData(x, old, this, rgt);
	
	delete[] old;
	
#if DBG_SPLIT
	//std::cout<<" into "<<*this<<" and "<<*rgt;
#endif	
	connectToChildren();
	
	rgt->connectLeftChild(p.index);

	rgt->connectToChildren();
	
	Pair<KeyType, Entity> b;
	b.key = p.key;
	b.index = rgt;
	parentNode()->bounce(b);
}

template <typename KeyType> 
Pair<KeyType, Entity> BNode<KeyType>::partData(Pair<KeyType, Entity> x, Pair<KeyType, Entity> old[], BNode * lft, BNode * rgt, bool doSplitLeaf)
{
	Pair<KeyType, Entity> res, q;
	BNode * dst = rgt;
	
	int numKeysRight = 0;
	bool inserted = false;
	int i = MaxNumKeysPerNode - 1;

	for(;i >= 0; i--) {
		if(x.key > old[i].key && !inserted) { //std::cout<<" k"<<i<<" "<<old[i].key;
			q = x;
			i++;
			inserted = true;
		}
		else
			q = old[i];
			
		numKeysRight++;
		
		if(numKeysRight == MaxNumKeysPerNode / 2 + 1) {
			if(doSplitLeaf)
				dst->insertData(q);
				
			dst = lft;
			res = q;
		}
		else
			dst->insertData(q);
	}
	
	if(!inserted)
		dst->insertData(x);
			
	return res;
}

template <typename KeyType> 
bool BNode<KeyType>::leafBalance()
{
	if(leafBalanceRight()) return true;
	return	false;//leafBalanceLeft();
}

template <typename KeyType> 
bool BNode<KeyType>::leafBalanceRight()
{
    BNode * rgt = siblingNode();
	if(!rgt) return false;
	
	std::cout<<"\n balance ";
	
	const Pair<KeyType, Entity> s = rgt->firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s.key, found);
	
	if(!found) return false;
	dbgRightUp(rgt);

#if 0
	int n = countBalance(numKeys(), rgt->numKeys() );
	if(n<1) return false;

	KeyType kpop = rgt->firstData().key;
	Pair<KeyType, Entity> k; 
	
	std::cout<<" nb"<<n;
	for(int i = 0; i<n; ++i) {
	    k = rgt->firstData();
	    
	   insertData(k);
	    
	   rgt->removeKey(rgt->firstData().key);
	
	    //std::cout<<"\n aft "<<rgt->leftChildNode()->str();
	}
	crossed->replaceKey(kpop, rgt->firstData().key);
#else
	
	int n = shouldBalance(this, rgt);
	if(n == 0) return false;
	
	Pair<KeyType, Entity> old = rgt->firstData();
	if(n < 0) sendDataRight(-n, rgt);
	else siblingNode()->sendDataLeft(n, this);
	
	crossed->replaceKey(old.key, rgt->firstData().key);
#endif	
	std::cout<<"\n aft ";
	dbgRightUp(rgt);

	return true;
}

template <typename KeyType> 
bool BNode<KeyType>::leafBalanceLeft()
{
	const Pair<KeyType, Entity> s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s.key, found);
	
	if(!found) return false;
	
	BNode * leftSibling = crossed->leafLeftTo(s.key);
	
	if(leftSibling == this) return false;
	
	int k = shouldBalance(leftSibling, this);
	if(k == 0) return false;
	
	Pair<KeyType, Entity> old = firstData();
	if(k < 0) leftSibling->sendDataRight(-k, this);
	else this->sendDataLeft(k, leftSibling);
	
	crossed->replaceKey(old.key, firstData().key);
	
	//std::cout<<"\nbalanced "<<*leftSibling<<*this;
	return true;
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::ancestor(const KeyType & x, bool & found) const
{
    if(isRoot()) return NULL;
	if(parentNode()->hasKey(x)) {
		found = true;
		return parentNode();
	}
	
	return parentNode()->ancestor(x, found);
}

template <typename KeyType> 
bool BNode<KeyType>::hasKey(const KeyType & x) const
{
	if(numKeys() < 1) return firstKey() == x;
    if(x > lastKey() || x < firstKey() ) return false;
	return (findKey(x).found > -1);
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::leftTo(const KeyType & x) const
{
	if(numKeys()==1) return static_cast<BNode *>(leftChild());
	int i = keyLeft(x);
	if(i < 0) return static_cast<BNode *>(leftChild());
	return static_cast<BNode *>(index(i));
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::rightTo(const KeyType & x, Pair<KeyType, Entity> & k) const
{
	int ii = keyRight(x);
	if(ii > -1) {
		k = m_data[ii];
		return static_cast<BNode *>(m_data[ii].index);
	}
	return NULL;
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::leafLeftTo(const KeyType & x)
{
	if(isLeaf()) return this;
	
	BNode * n = leftTo(x);
	return n->leafLeftTo(x);
}

template <typename KeyType> 
void BNode<KeyType>::sendDataRight(int num, BNode * rgt)
{
	for(int i = 0; i < num; i++) {
		rgt->insertData(lastData());
		removeLastData();
	}
}

template <typename KeyType> 
void BNode<KeyType>::sendDataLeft(int num, BNode * lft)
{
/// if send all in leaf, connect sibling to lft
    if(num == numKeys() && isLeaf()) {
		Entity * rgt = sibling();
		lft->connectSibling(rgt);
	}
    
	for(int i = 0; i < num; i++) {
		lft->insertData(firstData());
		removeFirstData();
	}
}

template <typename KeyType> 
Pair<KeyType, Entity> BNode<KeyType>::lastData() const 
{ return m_data[numKeys() - 1]; }

template <typename KeyType> 
Pair<KeyType, Entity> BNode<KeyType>::firstData() const 
{ return m_data[0]; }

template <typename KeyType> 
void BNode<KeyType>::removeLastData()
{
	m_data[numKeys() - 1].index = NULL;
	reduceNumKeys();
}

template <typename KeyType> 
void BNode<KeyType>::removeFirstData()
{
    if(numKeys() == 1) {
        m_data[0].index = NULL;
        reduceNumKeys();
        return;
    }
    
	for(int i = 0; i < numKeys() - 1; i++) {
		m_data[i] = m_data[i+1];
	}
	m_data[numKeys() - 1].index = NULL;
	reduceNumKeys();
}

template <typename KeyType> 
void BNode<KeyType>::replaceKey(KeyType x, KeyType y)
{
	SearchResult s = findKey(x);
	if(s.found > -1) m_data[s.found].key = y;
}

template <typename KeyType> 
void BNode<KeyType>::replaceIndex(int n, Pair<KeyType, Entity> x)
{
	m_data[n].index = x.index;
}

template <typename KeyType> 
void BNode<KeyType>::removeLeaf(const KeyType & x)
{
	if(!removeDataLeaf(x)) {
		std::cout<<"\n error bnode cannot remove data "<<x
			<<" in "<<str();
		return;
	}
	if(!underflow()) return;

	if(!leafBalance())
	    mergeLeaf();
}

template <typename KeyType> 
bool BNode<KeyType>::mergeLeaf()
{
	if( mergeLeafRight() ) return true;
	return mergeLeafLeft();
}

template <typename KeyType> 
bool BNode<KeyType>::mergeLeafRight()
{
	Entity * rgt = sibling();
/// must have sibling
	if(!rgt) return false;
	
	BNode * up = siblingNode()->parentNode();
/// must share parent
	if(parentNode() != up) return false;
	
	if(!shouldMerge(this, siblingNode() )) return false;
	
	std::cout<<"\n\n merge ";
	dbgRightUp(siblingNode() );
	parentNode()->dbgLinks(false);
    
	Pair<KeyType, Entity> rgt1st = siblingNode()->firstData();

	siblingNode()->sendDataLeft(siblingNode()->numKeys(), this);

	delete rgt;
	
	std::cout<<"\n aft "<<str();

    Pair<KeyType, Entity> k = up->dataRightTo(rgt1st.key);
	
	k.index = this;
	int ki = up->keyRight(k.key);
	
	up->setData(ki, k);
	
	std::cout<<"\n set key "<<ki<<" "<<k.key
	<<"\n"<<up->str();
	
	up->pop(k);
	
	dbgRightUp(siblingNode() );
    
	return true;
}

template <typename KeyType> 
bool BNode<KeyType>::mergeLeafLeft()
{
	const Pair<KeyType, Entity> s = firstData();
/// must share parent
	if(!parentNode()->hasKey(s.key)) return false;
	BNode * leftSibling = parentNode()->leafLeftTo(s.key);
	return leftSibling->mergeLeafRight();
}

template <typename KeyType> 
bool BNode<KeyType>::removeDataLeaf(const KeyType & x)
{
    SearchResult s = findKey(x);
	if(s.found < 0) return false;
	
	int found = s.found;
    
/// remove leaf connection
	if(m_data[found].index) {
		delete m_data[found].index;
		m_data[found].index = NULL;
	}

/// last is easy
	if(found == numKeys() - 1) {
		reduceNumKeys();//std::cout<<"reduce last in leaf to "<<numKeys();
		return true;
	}
	
	for(int i= found; i < numKeys() - 1; i++)
        m_data[i] = m_data[i+1];
		
	if(found == 0) {
	    bool c = false;
		BNode * crossed = ancestor(x, c);
		if(c) crossed->replaceKey(x, firstData().key);
	}
		
    reduceNumKeys();//std::cout<<"reduce in leaf to "<<numKeys();
    return true;
}

template <typename KeyType> 
void BNode<KeyType>::behead()
{
	connectSibling(index(0) );
	for(int i= 0; i < numKeys() - 1; i++)
		m_data[i] = m_data[i+1];
	reduceNumKeys();
}

template <typename KeyType> 
void BNode<KeyType>::removeFirstData1()
{
    for(int i= 0; i < numKeys() - 1; i++)
		m_data[i] = m_data[i+1];
	reduceNumKeys();
}

template <typename KeyType> 
bool BNode<KeyType>::removeKey(const KeyType & x)
{
	SearchResult s = findKey(x);
	
	if(s.found < 0) return false;
	
	int found = s.found;
	
	if(found == 0) {
	    connectLeftChild(m_data[found].index);
	}
	else {
		m_data[found - 1].index = m_data[found].index;
	}

	if(found == numKeys() - 1) {
		reduceNumKeys();
		return true;
	}
	
	for(int i= found; i < numKeys() - 1; i++)
		m_data[i] = m_data[i+1];
		
    reduceNumKeys();
	return true;
}

template <typename KeyType> 
bool BNode<KeyType>::removeKeyAndData(const KeyType & x)
{
	SearchResult s = findKey(x);
	
	if(s.found < 0)
	    return false;
	
	int found = s.found;
	
	if(m_data[found].index) {
	     delete m_data[found].index;
	     m_data[found].index = 0;
	}

	if(found == numKeys() - 1) {
		reduceNumKeys();
		return true;
	}
	
	for(int i= found; i < numKeys() - 1; i++)
		m_data[i] = m_data[i+1];
		
    reduceNumKeys();
	return true;
}

template <typename KeyType> 
void BNode<KeyType>::pop(const Pair<KeyType, Entity> & x)
{
	if(isRoot() ) popRoot(x);
	else popInterior(x);
}

template <typename KeyType> 
void BNode<KeyType>::popRoot(const Pair<KeyType, Entity> & x)
{
	std::cout<<"\n pop in "<<str()<<" by "<<x.key;
	removeKey(x.key);
	
	if(!underflow() ) return;
	
	if(hasChildren() ) {
	    
		BNode * lft = leftChildNode();
        setNumKeys(0);
		mergeData(lft);
		
		std::cout<<"\n merge 1st "<<str();
		
		if(lft->isLeaf() )
			connectLeftChild(NULL);
		else 
			connectLeftChild(lft->leftChild());

		connectToChildren();
		
		std::cout<<"\n done";
	}
	else {
	    connectLeftChild(NULL);
	}
	
	std::cout<<"\n aft pop "<<str();
	dbgLinks();
}

template <typename KeyType> 
void BNode<KeyType>::popInterior(const Pair<KeyType, Entity> & x)
{
	std::cout<<"\n pop by "<<x.key;
	dbgUp();
	
	if(x.key == firstKey() ) {
	    std::cout<<"\n first key";
	    removeFirstData1();
	}
	else removeKey(x.key);
	
	// validateKeys();
	std::cout<<"\n aft ";
	dbgUp();
	dbgLinks(false);
	
	if(!underflow()) return;
	
	//if(!interiorBalance())
	mergeInterior(x.key);
	
}

template <typename KeyType> 
void BNode<KeyType>::mergeData(BNode * another, int start)
{
	const int num = another->numKeys();
	for(int i = start; i < num; i++)
		insertData(another->data(i));
}

template <typename KeyType> 
const Pair<KeyType, Entity> BNode<KeyType>::dataRightTo(const KeyType & x) const
{
	int i = keyRight(x);
	if(i < 0) return lastData();
	return m_data[i];
}

template <typename KeyType> 
bool BNode<KeyType>::mergeInterior(const KeyType x)
{
	if(mergeInteriorRight(x)) return true;
	return mergeInteriorLeft(x);
}

template <typename KeyType> 
bool BNode<KeyType>::interiorBalance()
{
    bool stat = interiorBalanceRight();
	return stat;//interiorBalanceLeft();
}

template <typename KeyType> 
bool BNode<KeyType>::mergeInteriorRight(const KeyType x)
{
    int kr;
    BNode * rgt = rightInteriorNeighbor(kr);
	
    if(!rgt) return false;
    
	std::cout<<"\n  merge "<<str()<<" by "<<x;
	dbgRightUp(rgt);
	parentNode()->dbgLinks(false);
	
	if(!shouldInteriorMerge(this, rgt)) {
	    std::cout<<"\n  inner cannot merge";
	    return false;
	}
	
				
	std::cout<<"\n bring down k["<<kr<<"] "<<parentNode()->key(kr);
	Pair<KeyType, Entity> k;
	k.key = parentNode()->key(kr);
	k.index = rgt->leftChild();
	std::cout<<" <- right node "<<rgt->str();
	
	insertData(k);
    
	rgt->sendDataLeft(rgt->numKeys(), this);
    
	std::cout<<"\n aft "<<str();
	
	connectToChildren();
	
	std::cout<<"\n parent key right to last "<<kr<<" k "<<k.key;
	k.key = rgt->firstKey();
	k.index = this;
	parentNode()->setData(kr, k);
	
	std::cout<<"\n parent aft set k "<< parentNode()->str();
	
	parentNode()->pop(k);
	
	std::cout<<"\n parent aft pop k "<< parentNode()->str();
	
	return true;
}

template <typename KeyType> 
bool BNode<KeyType>::mergeInteriorLeft(const KeyType x)
{
	BNode * lft =leftInteriorNeighbor();
	if(!lft) return false;
	
	return lft->mergeInteriorRight(x);
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::rightInteriorNeighbor(int & indParent) const
{
    if(parentNode()->lastKey() < firstKey() ) return NULL;
    if(parentNode()->firstKey() >= lastKey() ) {
        indParent = 0;
        std::cout<<"\n right inner nei kr "<<indParent;
        return parentNode()->linkedNode(0);
    }
    
    SearchResult s = parentNode()->findKey(firstKey() );
	if(s.found > -1) indParent = s.found;
	else indParent = s.low;
	
	std::cout<<"\n right inner nei kr "<<indParent;
	
	return parentNode()->linkedNode(indParent);
}

/// parent.index[k] -> key[0]
/// j <- k-1
/// parent.index[j] -> left_neighbor.key[0]
template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::leftInteriorNeighbor() const
{
    if(parentNode()->firstKey() > lastKey() ) return NULL;
    if(parentNode()->index(0) == this) return parentNode()->leftChildNode();
    
	Pair<KeyType, Entity> k, j;
	if(!parentNode()->dataLeftTo(firstKey(), k)) return NULL;
	j = k;
	if(j.index == this)
		parentNode()->dataLeftTo(k.key, j);
	
	if(j.index == this) return NULL;
	return static_cast<BNode *>(j.index);
}

template <typename KeyType> 
void BNode<KeyType>::setData(int k, const Pair<KeyType, Entity> & x)
{ m_data[k] = x; }

template <typename KeyType> 
int BNode<KeyType>::keyRight(const KeyType & x) const
{
	if(lastKey() < x) return -1;
	SearchResult s = findKey(x);
	if(s.found > -1) return s.found;
	return s.high;
}

template <typename KeyType> 
int BNode<KeyType>::keyLeft(const KeyType & x) const
{
	if(lastKey() < x) return numKeys() - 1;
	if(firstKey() >= x) return -1;
	SearchResult s = findKey(x);
	int ii = s.low;
	if(s.found > -1) ii = s.found - 1;
	else if(key(s.high) < x) ii = s.high;
	
	return ii;
}

template <typename KeyType> 
bool BNode<KeyType>::dataLeftTo(const KeyType & x, Pair<KeyType, Entity> & dst) const
{
	int i = keyLeft(x);
	if(i > -1) {
		dst = m_data[i];
		return true;
	}
	dst.index = leftChild();
	return false;
}

template <typename KeyType> 
void BNode<KeyType>::dbgUp()
{
    int kr = parentNode()->keyLeft(firstKey() );
    if(kr < 0) kr = 0;
	std::cout<<"\n parent "<<parentNode()->str()
	    <<"\n k "<<kr
	    <<"\n  "<<str();
}

template <typename KeyType> 
void BNode<KeyType>::dbgRightUp(BNode<KeyType> * rgt)
{
    int kr = parentNode()->keyLeft(rgt->firstKey() );
    if(kr < 0) kr = 0;
	std::cout<<"\n parent "<<parentNode()->str()
	    <<"\n k "<<kr
	    <<"\n  "<<str()
	    <<" <- "<<rgt->str();
}

template <typename KeyType> 
bool BNode<KeyType>::interiorBalanceRight()
{ 
    int kr;
    BNode * rgt = rightInteriorNeighbor(kr);
	if(!rgt) return false;
	
	std::cout<<"\n\n balance "<<str()
	<<"->"<<rgt->str();
	
	int n = countBalance(numKeys(), rgt->numKeys() );
	if(n<1) return false;

	Pair<KeyType, Entity> k;
	
	for(int i = 0; i<n; ++i) {
		k.key = rgt->firstKey();
		k.index = rgt->leftChild();
	    
	    insertData(k);
	    
	    rgt->behead();
	
	    std::cout<<"\n aft "<<rgt->leftChildNode()->str();
	}
	
	k.key = rgt->leftChildNode()->firstKey();
	k.index = rgt;
	parentNode()->setData(kr, k);
	
	validateKeys();
	rgt->validateKeys();
	
	std::cout<<"\n aft "<<str()
	<<"->"<<rgt->str();
	
	return true;
}

template <typename KeyType> 
bool BNode<KeyType>::interiorBalanceLeft()
{
	BNode * lft = leftInteriorNeighbor();
	if(!lft) return false;
	Pair<KeyType, Entity> k;
	parentNode()->rightTo(lft->lastData().key, k);
	k.index = leftChild();
	insertData(k);
	Pair<KeyType, Entity> l = lft->lastData();
	connectLeftChild(l.index);
	parentNode()->replaceKey(k.key, l.key);
	lft->removeLastData();
	return true;
}

template <typename KeyType>
BNode<KeyType> * BNode<KeyType>::linkedNode(int idx) const
{ return static_cast<BNode *>(index(idx) ); }

template <typename KeyType>
BNode<KeyType> * BNode<KeyType>::leftChildNode() const
{ return static_cast<BNode *>(leftChild() ); }

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::parentNode() const
{ return static_cast<BNode *>(parent()); }

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::siblingNode() const
{ return static_cast<BNode *>(sibling()); }

template <typename KeyType> 
const std::string BNode<KeyType>::str() const 
{
	std::stringstream sst;
	if(isRoot() ) sst<<" root  ";
	else if(isLeaf() ) sst<<" leaf  ";
	else sst<<" inner ";
	
	int i = 0;
	sst<<" n "<<numKeys()<<" [";
    if(numKeys() ==0) {
        sst<<"] ";
    } else {
    
    for(;i<numKeys()-1;++i) sst<<key(i)<<",";
    sst<<lastKey()<<"] ";
    }
	
	if(!siblingNode() ) sst<<"~";
	return sst.str();
}

template <typename KeyType> 
void BNode<KeyType>::dbgDown() const 
{
	int i;
	if(isLeaf() ) {
		if(siblingNode() ) std::cout<<" | "<<siblingNode()->firstKey();
		else std::cout<<"~";
	}
	else {
		if(leftChildNode() ) std::cout << "\n   " << leftChildNode()->firstKey() << " < ";
		for(i=0;i<numKeys();++i)
		std::cout << " > " << linkedNode(i)->firstKey();
	}
}

template <typename KeyType>
bool BNode<KeyType>::isKeyInRange(const KeyType & x) const
{ 
	if( x < firstKey() ) return false;
	if( x > lastKey() ) return false; 
	return true;
}

template <typename KeyType> 
const SearchResult BNode<KeyType>::findKey(const KeyType & x) const
{
	SearchResult r;
    r.found = -1;
    r.low = 0; 
	r.high = 0;

	if(numKeys() < 1) return r;
	
	if(key(0) == x) r.found = 0;
	
	r.high = numKeys() - 1;
	if(key(r.high) == x) r.found = r.high;
	
    int mid;
    while(r.low < r.high - 1) {
        mid = (r.low + r.high) / 2;
        
		if(key(mid) == x) r.found = mid;
        else if(key(r.low) == x) r.found = r.low;
		else if(key(r.high) == x) r.found = r.high;
		
        if(r.found > -1) break;
		
        if(x < key(mid)) r.high = mid;
        else r.low = mid;
    }
    
    return r;
}

template <typename KeyType> 
Pair<Entity *, Entity> BNode<KeyType>::find(const KeyType & x)
{
	if(isRoot()) 
		return findRoot(x);
	else if(isLeaf())
		return findLeaf(x);
	
	return findInterior(x);
}

template <typename KeyType> 
Pair<Entity *, Entity> BNode<KeyType>::findRoot(const KeyType & x)
{
	if(hasChildren()) {
		BNode * n = nextIndex(x);
		return n->find(x);
	}
	return findLeaf(x);
}

template <typename KeyType> 
Pair<Entity *, Entity> BNode<KeyType>::findLeaf(const KeyType & x)
{
	Pair<Entity *, Entity> r;
	r.key = this;
	r.index = NULL;
	if(!isKeyInRange(x) ) {
		return r;
	}
	int f = findKey(x).found;
	if(f < 0) return r;
	
	r.index = dataP(f)->index;
	
	return r;
}

template <typename KeyType> 
Pair<Entity *, Entity> BNode<KeyType>::findInterior(const KeyType & x)
{
	BNode * n = nextIndex(x);
	return n->find(x);
}

template <typename KeyType>
Pair<Entity *, Entity> BNode<KeyType>::findInNode(const KeyType & x, SearchResult * result)
{ 
	Pair<Entity *, Entity> r;
	r.key = this;
	r.index = NULL;
	if(!isKeyInRange(x)) {
		result->found = -1;
		result->low = 0;
		result->high = numKeys() - 1;
		if(firstKey() > x) r.index = firstData().index;
		else r.index = lastData().index;
		return r;
	}
	
	*result = findKey(x);

	if(result->found < 0) return r;
	
	r.index = dataP(result->found)->index;
	
	return r; 
}

template <typename KeyType>
void BNode<KeyType>::deepPrint() const
{
	std::cout<<"\n "<<*this<<":";
	std::cout<<" "<<*leftTo(key(0));
	int i=0;
	Pair<KeyType, Entity> k;
	for(;i<numKeys();++i) std::cout<<" "<<key(i)<<":"<<*rightTo(key(i), k);
}

template <typename KeyType>
bool BNode<KeyType>::checkDupKey(KeyType & duk) const
{
	KeyType k0 = key(0);
	int i=1;
	for(;i<numKeys();++i) {
		if(key(i) > k0) 
			k0 = key(i);
		else {
			std::cout<<"\n k"<<i<<" "<<key(i)<<">="<<k0<<" ";
			duk = key(i);
			return false;
		}
	}
	return true;
}

template <typename KeyType>
void BNode<KeyType>::dbgFind(const KeyType & x)
{
	if(isRoot()) 
		return dbgFindInRoot(x);
	else if(isLeaf())
		return dbgFindInLeaf(x);
	
	return dbgFindInInterior(x);
}

template <typename KeyType>
void BNode<KeyType>::dbgFindInRoot(const KeyType & x)
{
	std::cout<<"\n dbg root "<<str();
	if(hasChildren()) {
		BNode * n = nextIndex(x);
		return n->dbgFind(x);
	}
	return dbgFindInLeaf(x);
}

template <typename KeyType>
void BNode<KeyType>::dbgFindInLeaf(const KeyType & x)
{
	std::cout<<"\n dbg leaf "<<str();
	if(!isKeyInRange(x) ) {
		std::cout<<"\n out of range";
		return;
	}
	int found = findKey(x).found;
	
	if(found < 0) {
		std::cout<<"\n not found";
		return;
	}
	
	std::cout<<"\n found "<<x;
}

template <typename KeyType>
void BNode<KeyType>::dbgFindInInterior(const KeyType & x)
{
	std::cout<<"\n dbg inner "<<str();
	BNode * n = nextIndex(x);
	return n->dbgFind(x);
}

template <typename KeyType>
bool BNode<KeyType>::dbgLinks(bool silient)
{	
    if(!hasChildren()) return true;
	bool stat = true;
	if(!silient) std::cout<<"\n n k "<<numKeys();
	KeyType pre = leftChildNode()->firstKey();
	if(!silient) std::cout<<" "<<pre<<" | ";
	
	for(int i=0;i<numKeys(); ++i) {
	
		KeyType cur = linkedNode(i)->firstKey();
		if(!silient) std::cout<<"  "<<cur;
		if(cur <= pre) {
			std::cout<<"\n\n****    error wrong link "
			        <<pre<<" >= "<<cur;
			stat = false;
		}
		pre = cur;
	}
	
	for(int i=0;i<numKeys(); ++i) {
		if(key(i) > linkedNode(i)->firstKey() ) {
			if(!silient) std::cout<<"\n\n****    error wrong k["<<i<<"]"
						<<key(i)
						<<" > "
						<<linkedNode(i)->str()
						;
			stat = false;
		}
	}
	
	if(isLeaf() ) {
		if(siblingNode() ) {
			if(lastKey() >= siblingNode()->firstKey() ) {
				if(!silient) std::cout<<"\n\n****    error order to sibling "
							<<lastKey()
							<<" > "
							<<siblingNode()->firstKey()
							;
				stat = false;
			}
		}
	}
	else if(leftChild() ) {
		if(firstKey() < leftChildNode()->lastKey() ) {
				if(!silient) std::cout<<"\n\n****    error order to left child "
							<<firstKey()
							<<" < ["<<leftChildNode()->firstKey()
							<<", "<<leftChildNode()->lastKey()<<"] "
							;
				stat = false;
			}
	}
	if(!stat) {
	    std::cout<<"\n failed link check\n"<<str();
	    dbgDown();
	}
	return stat;
}

template <typename KeyType>
int BNode<KeyType>::countBalance(int a, int b)
{ return (a + b)/ 2 - a;}

template <typename KeyType>
bool BNode<KeyType>::shouldMerge(BNode<KeyType> * lft, BNode<KeyType> * rgt) const 
{ return (lft->numKeys() + rgt->numKeys()) < MaxNumKeysPerNode; }

template <typename KeyType>
bool BNode<KeyType>::isFull() const 
{ return m_numKeys == MaxNumKeysPerNode; }
	
template <typename KeyType>
bool BNode<KeyType>::underflow() const 
{ return m_numKeys <= MinNumKeysPerNode; }

template <typename KeyType>
bool BNode<KeyType>::shouldInteriorMerge(BNode<KeyType> * lft, BNode<KeyType> * rgt) const 
{ return (lft->numKeys() + rgt->numKeys()) < MaxNumKeysPerNode; }
	
template <typename KeyType>
int BNode<KeyType>::shouldBalance(BNode<KeyType> * lft, BNode<KeyType> * rgt) const 
{ return (lft->numKeys() + rgt->numKeys()) / 2 - lft->numKeys(); }
	
template <typename KeyType>
void BNode<KeyType>::reduceNumKeys() 
{ m_numKeys--; }
	
template <typename KeyType>
void BNode<KeyType>::increaseNumKeys() 
{ m_numKeys++; }
	
template <typename KeyType>
void BNode<KeyType>::setNumKeys(int x) 
{ m_numKeys = x; }

template <typename KeyType>
void BNode<KeyType>::validateKeys()
{
    /*for(int i=0;i<numKeys(); ++i) {
		if(linkedNode(i)->firstKey() < key(i)) {
			m_data[i].key = linkedNode(i)->firstKey();
		}
	}*/
	
	for(int i=0; i< numKeys()-1; ++i) {
	    if(index(i) == index(i+1)) {
	        
	        for(int j=i;j<numKeys()-1;++j) {
	            m_data[j] = m_data[j+1];
	        }
	        m_numKeys--;
	    }
	}
}
	
} // end of namespace sdb

}
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

namespace sdb {

class TreeNode : public Entity
{
public:
	TreeNode(Entity * parent = NULL);
	virtual ~TreeNode();
	
	bool isRoot() const;
	bool hasChildren() const;
	bool isLeaf() const;
	
	Entity * sibling() const;
	Entity * firstIndex() const;
	
	void setLeaf();
	void connectSibling(Entity * another);
	void setFirstIndex(Entity * another);
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
	static NodeIndirection SeparatedNodes;
private:
	Entity *m_first;
	bool m_isLeaf;
};

class SearchResult
{
public:
	int found, low, high;
};

template <typename KeyType>
class BNode : public TreeNode
{
public:
	BNode(Entity * parent = NULL);
	virtual ~BNode();
	
	Pair<KeyType, Entity> * insert(const KeyType & x);
    void remove(const KeyType & x);
	Pair<Entity *, Entity> find(const KeyType & x);
	
    void getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const;
	BNode * firstLeaf();
	BNode * nextLeaf() { return static_cast<BNode *>(sibling()); }
	
	friend std::ostream& operator<<(std::ostream &output, const BNode & p) {
        output << p.str();
        return output;
    }
	
	Pair<Entity *, Entity> findInNode(const KeyType & x);
	
	const int numKeys() const  { return m_numKeys; }
	const KeyType key(const int & i) const { return m_data[i].key; }
	Entity * index(const int & i) const { return m_data[i].index; }
	static SearchResult LatestSearch;
private:	
	const KeyType firstKey() const;
	const KeyType lastKey() const;
	
    BNode *nextIndex(KeyType x) const;
	
	Pair<KeyType, Entity> * insertRoot(const KeyType & x);
	Pair<KeyType, Entity> * insertLeaf(const KeyType & x);
	Pair<KeyType, Entity> * insertInterior(const KeyType & x);
	
	BNode *splitRoot(KeyType x);
	BNode *splitLeaf(const KeyType & x);
	
	void insertData(Pair<KeyType, Entity> x);
	
	void insertKey(KeyType x);
	bool removeKey(const KeyType & x);
	bool removeKeyAndData(const KeyType & x);

	void partRoot(Pair<KeyType, Entity> x);
	Pair<KeyType, Entity> partData(Pair<KeyType, Entity> x, Pair<KeyType, Entity> old[], BNode * lft, BNode * rgt, bool doSplitLeaf = false);
	
	void partInterior(Pair<KeyType, Entity> x);
	
	void connectChildren();
	
	void bounce(Pair<KeyType, Entity> b);

	void balanceLeaf();
	bool balanceLeafRight();
	void balanceLeafLeft();
	void rightData(int num, BNode * rgt);
	void leftData(int num, BNode * lft);
	
	Pair<KeyType, Entity> lastData() const;
	Pair<KeyType, Entity> firstData() const;
	
	void removeLastData();
	void removeFirstData();

	void replaceKey(KeyType x, KeyType y);
	
	BNode * ancestor(const KeyType & x, bool & found) const;
	BNode * leftTo(const KeyType & x) const;
	BNode * rightTo(const KeyType & x, Pair<KeyType, Entity> & k) const;
	BNode * leafLeftTo(const KeyType & x);
	
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
	
	const Pair<KeyType, Entity> data(int x) const { return m_data[x]; }
	Pair<KeyType, Entity> * dataP(int x) const { return &m_data[x]; }
	
	void mergeData(BNode * another, int start = 0);
	void replaceIndex(int n, Pair<KeyType, Entity> x);
	
	const Pair<KeyType, Entity> dataRightTo(const KeyType & x) const;
	
	bool mergeInterior();
	void balanceInterior();
	
	bool mergeInteriorRight();
	bool mergeInteriorLeft();
	
	void setData(int k, const Pair<KeyType, Entity> & x);
	
	bool dataLeftTo(const KeyType & x, Pair<KeyType, Entity> & dst) const;
	
	BNode * leftInteriorNeighbor() const;
	
	bool balanceInteriorRight();
	void balanceInteriorLeft();
	
	BNode * parentNode() const;
	
	bool isFull() const { return m_numKeys == MaxNumKeysPerNode; }
	bool underflow() const { return m_numKeys < MinNumKeysPerNode; }
	
	void reduceNumKeys() { m_numKeys--; }
	void increaseNumKeys() { m_numKeys++; }
	void setNumKeys(int x) { m_numKeys = x; }
	
	bool shouldMerge(BNode * lft, BNode * rgt) const { return (lft->numKeys() + rgt->numKeys()) <= MaxNumKeysPerNode; }
	bool shouldInteriorMerge(BNode * lft, BNode * rgt) const { return (lft->numKeys() + rgt->numKeys()) < MaxNumKeysPerNode; }
	int shouldBalance(BNode * lft, BNode * rgt) const { return (lft->numKeys() + rgt->numKeys()) / 2 - lft->numKeys(); }
	
	bool isKeyInRange(const KeyType & x) const;
	const SearchResult findKey(const KeyType & x) const;
	int keyRight(const KeyType & x) const;
	int keyLeft(const KeyType & x) const;
	
	Pair<Entity *, Entity> findRoot(const KeyType & x);
	Pair<Entity *, Entity> findLeaf(const KeyType & x);
	Pair<Entity *, Entity> findInterior(const KeyType & x);
		
	const std::string str() const;
private:
    Pair<KeyType, Entity> * m_data;
	int m_numKeys;
};

template <typename KeyType>
SearchResult BNode<KeyType>::LatestSearch;

template <typename KeyType>  
BNode<KeyType>::BNode(Entity * parent) : TreeNode(parent)
{
	m_numKeys = 0;
	m_data = new Pair<KeyType, Entity>[MaxNumKeysPerNode];
	for(int i=0;i< MaxNumKeysPerNode;i++)
        m_data[i].index = NULL;
}

template <typename KeyType> 
BNode<KeyType>::~BNode()
{
	for(int i=0;i< numKeys();i++) {
		if(m_data[i].index) delete m_data[i].index;
	}
	delete[] m_data;
}

template <typename KeyType>  
BNode<KeyType> * BNode<KeyType>::nextIndex(KeyType x) const
{
	if(firstKey() > x) return static_cast<BNode *>(firstIndex());
	int ii;
	SearchResult s = findKey(x);
	ii = s.low;
	if(s.found > -1) ii = s.found;
	else if(key(s.high) < x) ii = s.high;
	
	//std::cout<<"find "<<x<<" in "<<*this<<" i "<<ii<<"\n";
	return (static_cast<BNode *>(m_data[ii].index));
}

template <typename KeyType> 
const KeyType BNode<KeyType>::firstKey() const { return m_data[0].key; }

template <typename KeyType> 
const KeyType BNode<KeyType>::lastKey() const { return m_data[numKeys() - 1].key; }

template <typename KeyType> 
Pair<KeyType, Entity> * BNode<KeyType>::insert(const KeyType & x)
{
	if(isRoot()) 
		return insertRoot(x);
	else if(isLeaf())
		return insertLeaf(x);
	
	return insertInterior(x);
}

template <typename KeyType> 
void BNode<KeyType>::remove(const KeyType & x)
{
    if(isRoot()) {
    
		SeparatedNodes.reset();
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
		
		setFirstIndex(NULL);
	}
}

template <typename KeyType> 
Pair<KeyType, Entity> * BNode<KeyType>::insertRoot(const KeyType & x)
{
	if(hasChildren()) {
		BNode * n = nextIndex(x);
		return n->insert(x);
	}
	
	BNode * dst = this;
	if(!hasKey(x)) {
		if(isFull()) {
			dst = splitRoot(x);
		}
		else
			insertKey(x);
	}
	return dst->dataP(dst->findKey(x).found);
}

template <typename KeyType> 
Pair<KeyType, Entity> * BNode<KeyType>::insertLeaf(const KeyType & x)
{
	BNode * dst = this;
	if(!hasKey(x)) {
		if(isFull())
			dst = splitLeaf(x);
		else
			insertKey(x);
	}
	return dst->dataP(dst->findKey(x).found);
}

template <typename KeyType> 
Pair<KeyType, Entity> * BNode<KeyType>::insertInterior(const KeyType & x)
{
	BNode * n = nextIndex(x);
	return n->insert(x);
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
    for(i= numKeys() - 1;i >= 0 && m_data[i].key > x; i--)
        m_data[i+1] = m_data[i];
		
    m_data[i+1].key = x;
	m_data[i+1].index = NULL;
	//std::cout<<"insert key "<<x.key<<" at "<<i+1<<"\n";
    increaseNumKeys();
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::splitRoot(KeyType x)
{
	//std::cout<<"split root "<<*this;
	Entity * dangling = SeparatedNodes.give();
	BNode * one = static_cast<BNode *>(dangling);
	if(one) {
		one->setParent(this);
		one->setNumKeys(0);
		SeparatedNodes.reset();
	}
	else
		one = new BNode(this);
	
	//BNode * one = new BNode(this); 
	one->setLeaf();
	BNode * two = new BNode(this); two->setLeaf();
	
	Pair<KeyType, Entity> ex;
	ex.key = x;
	partData(ex, m_data, one, two, true);
	
	//std::cout<<"into "<<*one<<*two;
	
	setFirstIndex(one);
	m_data[0].key = two->firstKey();
	m_data[0].index = two;
	one->connectSibling(two);
	setNumKeys(1);
	one->balanceLeafLeft();
	
	if(one->hasKey(x)) return one;
	return two;
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::splitLeaf(const KeyType & x)
{
	//std::cout<<"split leaf "<<*this;
	//std::cout<<"into ";
	Entity * oldRgt = sibling();
	BNode * two = new BNode(parent()); two->setLeaf();

	Pair<KeyType, Entity> *old = new Pair<KeyType, Entity>[MaxNumKeysPerNode];
	for(int i=0; i < MaxNumKeysPerNode; i++)
		old[i] = m_data[i];
		
	setNumKeys(0);
	
	Pair<KeyType, Entity> ex;
	ex.key = x;
	partData(ex, old, this, two, true);
	//std::cout<<*this<<*two;
	delete[] old;
	connectSibling(two);
	if(oldRgt) two->connectSibling(oldRgt);
	
	Pair<KeyType, Entity> b;
	b.key = two->firstKey();
	b.index = two;
	parentNode()->bounce(b);
	//balanceLeafLeft();
	
	if(two->hasKey(x)) return two;
	return this;
}

template <typename KeyType> 
void BNode<KeyType>::bounce(Pair<KeyType, Entity> b)
{	
	//std::cout<<"bounce "<<b.key<<"\n";
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
	dst[level].push_back(firstIndex());
	for(int i = 0;i < numKeys(); i++)
		dst[level].push_back(index(i));
		
	level++;
		
	BNode * n = static_cast<BNode *>(firstIndex());
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
			return static_cast<BNode *>(firstIndex())->firstLeaf();
		else 
			return this;
			
	}
	
	if(isLeaf())
		return this;
	
	return static_cast<BNode *>(firstIndex())->firstLeaf();
}

template <typename KeyType> 
void BNode<KeyType>::connectChildren()
{
	if(!hasChildren()) return;
	Entity * n = firstIndex();
	n->setParent(this);
	for(int i = 0;i < numKeys(); i++) {
		n = index(i);
		n->setParent(this);
	}
}


template <typename KeyType> 
void BNode<KeyType>::partRoot(Pair<KeyType, Entity> x)
{
	//std::cout<<"part root "<<*this;
	BNode * one = new BNode(this);
	BNode * two = new BNode(this);
	
	Pair<KeyType, Entity> p = partData(x, m_data, one, two);
	
	//std::cout<<"into "<<*one<<*two;
	
	one->setFirstIndex(firstIndex());
	two->setFirstIndex(p.index);
	
	setFirstIndex(one);
	m_data[0].key = p.key;
	m_data[0].index = two;

	setNumKeys(1);
	
	one->connectChildren();
	two->connectChildren();
}

template <typename KeyType> 
void BNode<KeyType>::partInterior(Pair<KeyType, Entity> x)
{
	//std::cout<<"part interior "<<*this;
	
	BNode * rgt = new BNode(parent());
	
	Pair<KeyType, Entity> * old = new Pair<KeyType, Entity>[MaxNumKeysPerNode];
	for(int i=0; i < MaxNumKeysPerNode; i++)
		old[i] = m_data[i];
	
	setNumKeys(0);
	Pair<KeyType, Entity> p = partData(x, old, this, rgt);
	
	delete[] old;
	
	//std::cout<<"into "<<*this<<*rgt;
	
	connectChildren();
	
	rgt->setFirstIndex(p.index);
	rgt->connectChildren();
	
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
	for(int i = MaxNumKeysPerNode - 1;i >= 0; i--) {
		if(x.key > old[i].key && !inserted) {
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
void BNode<KeyType>::balanceLeaf()
{
	if(!balanceLeafRight()) {
		balanceLeafLeft();
	}
}

template <typename KeyType> 
bool BNode<KeyType>::balanceLeafRight()
{
	Entity * rgt = sibling();
	if(!rgt) return false;
	
	BNode * rgtNode = static_cast<BNode *>(rgt);
	const Pair<KeyType, Entity> s = rgtNode->firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s.key, found);
	
	if(!found) return false;
	
	int k = shouldBalance(this, rgtNode);
	if(k == 0) return false;
	
	Pair<KeyType, Entity> old = rgtNode->firstData();
	if(k < 0) rightData(-k, rgtNode);
	else rgtNode->leftData(k, this);
	
	crossed->replaceKey(old.key, rgtNode->firstData().key);
	
	return true;
}

template <typename KeyType> 
void BNode<KeyType>::balanceLeafLeft()
{
	const Pair<KeyType, Entity> s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s.key, found);
	
	if(!found) return;
	
	BNode * leftSibling = crossed->leafLeftTo(s.key);
	
	if(leftSibling == this) return;
	
	int k = shouldBalance(leftSibling, this);
	if(k == 0) return;
	
	Pair<KeyType, Entity> old = firstData();
	if(k < 0) leftSibling->rightData(-k, this);
	else this->leftData(k, leftSibling);
	
	crossed->replaceKey(old.key, firstData().key);
	
	//std::cout<<"\nbalanced "<<*leftSibling<<*this;
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::ancestor(const KeyType & x, bool & found) const
{
	if(parentNode()->hasKey(x)) {
		found = true;
		return parentNode();
	}
	
	if(parentNode()->isRoot()) return NULL;
	return parentNode()->ancestor(x, found);
}

template <typename KeyType> 
bool BNode<KeyType>::hasKey(const KeyType & x) const
{
    if(x > lastKey() || x < firstKey()) return false;
	return findKey(x).found > -1;
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::leftTo(const KeyType & x) const
{
	if(numKeys()==1) return static_cast<BNode *>(firstIndex());
	int i = keyLeft(x);
	if(i < 0) return static_cast<BNode *>(firstIndex());
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
void BNode<KeyType>::rightData(int num, BNode * rgt)
{
	for(int i = 0; i < num; i++) {
		rgt->insertData(lastData());
		removeLastData();
	}
}

template <typename KeyType> 
void BNode<KeyType>::leftData(int num, BNode * lft)
{
	for(int i = 0; i < num; i++) {
		lft->insertData(firstData());
		removeFirstData();
	}
}

template <typename KeyType> 
Pair<KeyType, Entity> BNode<KeyType>::lastData() const { return m_data[numKeys() - 1]; }

template <typename KeyType> 
Pair<KeyType, Entity> BNode<KeyType>::firstData() const { return m_data[0]; }

template <typename KeyType> 
void BNode<KeyType>::removeLastData()
{
	m_data[numKeys() - 1].index = NULL;
	reduceNumKeys();
}

template <typename KeyType> 
void BNode<KeyType>::removeFirstData()
{
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
	if(!removeDataLeaf(x)) return;
	if(!underflow()) return;
	
	if(!mergeLeaf())
		balanceLeaf();
}

template <typename KeyType> 
bool BNode<KeyType>::mergeLeaf()
{
	if(mergeLeafRight())
		return true;
		
	return mergeLeafLeft();
}

template <typename KeyType> 
bool BNode<KeyType>::mergeLeafRight()
{
	Entity * rgt = sibling();
	if(!rgt) return false;
	
	BNode * rgtNode = static_cast<BNode *>(rgt);
	
	const Pair<KeyType, Entity> s = rgtNode->firstData();
	
	if(!shouldMerge(this, rgtNode)) return false;
	
	BNode * up = rgtNode->parentNode();

	Pair<KeyType, Entity> old = rgtNode->firstData();
	Entity * oldSibling = rgtNode->sibling();
	
	rgtNode->leftData(rgtNode->numKeys(), this);
	
	delete rgt;
	
	connectSibling(oldSibling);
	
	Pair<KeyType, Entity> k = up->dataRightTo(old.key);
	
	if(parent() == up) {
		k.index = this;
		int ki = up->keyRight(k.key);
		up->setData(ki, k);
	}
	else {
		bool found = false;
		BNode * crossed = ancestor(s.key, found);
		crossed->replaceKey(s.key, k.key);
	}
	
	up->pop(k);
	
	return true;
}

template <typename KeyType> 
bool BNode<KeyType>::mergeLeafLeft()
{
	const Pair<KeyType, Entity> s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s.key, found);
	
	if(!found) return false;
	
	BNode * leftSibling = crossed->leafLeftTo(s.key);
	
	if(leftSibling == this) return false;
	
	return leftSibling->mergeLeafRight();
}

template <typename KeyType> 
bool BNode<KeyType>::removeDataLeaf(const KeyType & x)
{
	SearchResult s = findKey(x);
	if(s.found < 0) {
		// std::cout<<"cannot find key "<<x;
		return false;
	}
	
	int found = s.found;
	
	if(m_data[found].index) {
	    delete m_data[found].index;
		m_data[found].index = NULL;
	}
	
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
bool BNode<KeyType>::removeKey(const KeyType & x)
{
	SearchResult s = findKey(x);
	
	if(s.found < 0) { std::cout<<" cannot find key ";
	    return false;
	}
	
	int found = s.found;
	
	if(found == 0) {
	    setFirstIndex(m_data[found].index);
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
	
	if(s.found < 0) { std::cout<<" cannot find key ";
	    return false;
	}
	
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
	if(isRoot()) popRoot(x);
	else popInterior(x);
}

template <typename KeyType> 
void BNode<KeyType>::popRoot(const Pair<KeyType, Entity> & x)
{
	if(numKeys() > 1) {
	    const bool hc = hasChildren();
	    removeKey(x.key);
	    if(!hc) setFirstIndex(NULL);
	}
	else {
		BNode * lft = static_cast<BNode *>(firstIndex());
		setNumKeys(0);
		mergeData(lft);
		
		if(lft->isLeaf())
			setFirstIndex(NULL);
		else 
			setFirstIndex(lft->firstIndex());
			
		// delete lft;
		SeparatedNodes.take(lft);
		
		connectChildren();
	}
}

template <typename KeyType> 
void BNode<KeyType>::popInterior(const Pair<KeyType, Entity> & x)
{
	removeKey(x.key);
	if(!underflow()) return;
	//std::cout<<"interior underflow! "<<*this;
	if(!mergeInterior())
		balanceInterior();
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
bool BNode<KeyType>::mergeInterior()
{
	if(mergeInteriorRight()) return true;
	return mergeInteriorLeft();
}

template <typename KeyType> 
void BNode<KeyType>::balanceInterior()
{
	if(!balanceInteriorRight()) balanceInteriorLeft();
}



template <typename KeyType> 
bool BNode<KeyType>::mergeInteriorRight()
{
	Pair<KeyType, Entity> k;
	BNode * rgt = parentNode()->rightTo(lastData().key, k);
	if(!rgt) return false;
	if(!shouldInteriorMerge(this, rgt)) return false;
	
	k.index = rgt->firstIndex();
	
	insertData(k);
	
	rgt->leftData(rgt->numKeys(), this);
	
	// delete rgt;
	SeparatedNodes.take(rgt);
	
	connectChildren();
	
	int ki = parentNode()->keyRight(k.key); 
	k.index = this;
	parentNode()->setData(ki, k);
	
	parentNode()->pop(k);
	return true;
}

template <typename KeyType> 
bool BNode<KeyType>::mergeInteriorLeft()
{
	BNode * lft =leftInteriorNeighbor();
	if(!lft) return false;
	
	return lft->mergeInteriorRight();
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::leftInteriorNeighbor() const
{
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
{
	m_data[k] = x;
}

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
	dst.index = firstIndex();
	return false;
}

template <typename KeyType> 
bool BNode<KeyType>::balanceInteriorRight()
{
	Pair<KeyType, Entity> k;
	BNode * rgt = parentNode()->rightTo(lastData().key, k);
	if(!rgt) return false;

	k.index = rgt->firstIndex();
	
	insertData(k);
	
	parentNode()->replaceKey(k.key, rgt->firstData().key);
	
	rgt->removeKey(rgt->firstData().key);

	return true;
}

template <typename KeyType> 
void BNode<KeyType>::balanceInteriorLeft()
{
	BNode * lft = leftInteriorNeighbor();
	if(!lft) return;
	Pair<KeyType, Entity> k;
	parentNode()->rightTo(lft->lastData().key, k);
	k.index = firstIndex();
	insertData(k);
	Pair<KeyType, Entity> l = lft->lastData();
	setFirstIndex(l.index);
	parentNode()->replaceKey(k.key, l.key);
	lft->removeLastData();
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::parentNode() const
{
	return static_cast<BNode *>(parent());
}

template <typename KeyType> 
const std::string BNode<KeyType>::str() const 
{
	std::stringstream sst;
	sst.str("");
	int i;
    sst<<"(";
    for(i=0;i< numKeys();i++) {
		sst<<m_data[i].key;
		if(i< numKeys()-1) sst<<" ";
	}
    sst<<") ";
	return sst.str();
}

template <typename KeyType>
bool BNode<KeyType>::isKeyInRange(const KeyType & x) const
{ 
	if(numKeys() < 1) return false;
	return (x >= key(0) && x<= key(numKeys() - 1)); 
}

template <typename KeyType> 
const SearchResult BNode<KeyType>::findKey(const KeyType & x) const
{
	SearchResult & r = LatestSearch;
    r.found = -1;
    r.low = 0; 
	r.high = numKeys() - 1;
	if(numKeys() < 1) return r;
    int mid;
        
    while(r.low <= r.high) {
        mid = (r.low + r.high) / 2;
        
        if(key(r.low) == x) r.found = r.low;
        else if(key(r.high) == x) r.found = r.high;
        else if(key(mid) == x) r.found = mid;
        
        if(r.found > -1) break;
		
        if(x < key(mid)) r.high = mid;
        else r.low = mid;
        
		//std::cout<<" "<<r.low<<":"<<r.high<<"\n";
        if(r.low >= r.high - 1) break;
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
	if(!isKeyInRange(x)) return r;
	int found = findKey(x).found;
	
	if(found < 0) return r;
	r.index = data(found).index;
	return r;
}

template <typename KeyType> 
Pair<Entity *, Entity> BNode<KeyType>::findInterior(const KeyType & x)
{
	BNode * n = nextIndex(x);
	return n->find(x);
}

template <typename KeyType>
Pair<Entity *, Entity> BNode<KeyType>::findInNode(const KeyType & x)
{ return findLeaf(x); }

} // end of namespace sdb
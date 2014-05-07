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
	virtual ~TreeNode() {}
	
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
private:
	Entity *m_first;
	bool m_isLeaf;
};

template <typename KeyType, typename ValueType, class LeafType>
class BNode : public TreeNode
{
public:
	BNode(Entity * parent = NULL);
	virtual ~BNode();
	
	void insert(Pair<KeyType, ValueType> x);
    void remove(Pair<KeyType, ValueType> x);
	
    void getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const;
	BNode * firstLeaf();
	void getValues(std::vector<ValueType> & dst);
	
	friend std::ostream& operator<<(std::ostream &output, const BNode & p) {
        output << p.str();
        return output;
    }
	
	const int numKeys() const  { return m_numKeys; }
	const KeyType key(const int & i) const { return m_data[i].key; }
	
private:
	const KeyType firstKey() const;
	const KeyType lastKey() const;
	
    BNode *nextIndex(KeyType x) const;
	
	void insertRoot(Pair<KeyType, ValueType> x);
	BNode *splitRoot(KeyType x);
	
	void insertLeaf(Pair<KeyType, ValueType> x);
	BNode *splitLeaf(Pair<KeyType, ValueType> x);
	
	void insertData(Pair<KeyType, Entity> x);
	
	void insertKey(KeyType x);
	bool removeKey(const KeyType & x);

	void partRoot(Pair<KeyType, Entity> x);
	Pair<KeyType, Entity> partData(Pair<KeyType, Entity> x, Pair<KeyType, Entity> old[], BNode * lft, BNode * rgt, bool doSplitLeaf = false);
	
	void partInterior(Pair<KeyType, Entity> x);
	
	void insertInterior(Pair<KeyType, ValueType> x);
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
	BNode * leftTo(const Pair<KeyType, Entity> & x) const;
	BNode * rightTo(const Pair<KeyType, Entity> & x, Pair<KeyType, Entity> & k) const;
	BNode * leafLeftTo(Pair<KeyType, Entity> x);
	
	bool hasKey(const KeyType & x) const;
	
	void removeLeaf(const Pair<KeyType, ValueType> & x);
	bool removeDataLeaf(const Pair<KeyType, ValueType> & x);

	bool mergeLeaf();
	bool mergeLeafRight();
	bool mergeLeafLeft();
	
	void pop(const Pair<KeyType, Entity> & x);
	void popRoot(const Pair<KeyType, Entity> & x);
	void popInterior(const Pair<KeyType, Entity> & x);
	
	const Pair<KeyType, Entity> data(int x) const;
	void mergeData(BNode * another, int start = 0);
	void replaceIndex(int n, Pair<KeyType, Entity> x);
	
	const Pair<KeyType, Entity> dataRightTo(const Pair<KeyType, Entity> & x) const;
	
	bool mergeInterior();
	void balanceInterior();
	
	bool mergeInteriorRight();
	bool mergeInteriorLeft();
	
	void setData(int k, const Pair<KeyType, Entity> & x);
	int keyLeft(const KeyType & x) const;
	
	bool dataLeftTo(const Pair<KeyType, Entity> & x, Pair<KeyType, Entity> & dst) const;
	
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
	
	void setValue(const Pair<KeyType, ValueType> & x);
	
	int findKey(const KeyType & x) const;
	
	const std::string str() const;
private:
    Pair<KeyType, Entity> * m_data;
	int m_numKeys;
};

template <typename KeyType, typename ValueType, class LeafType>  
BNode<KeyType, ValueType, LeafType>::BNode(Entity * parent) : TreeNode(parent)
{
	m_numKeys = 0;
	m_data = new Pair<KeyType, Entity>[MaxNumKeysPerNode];
	for(int i=0;i< MaxNumKeysPerNode;i++)
        m_data[i].index = NULL;
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType>::~BNode()
{
    delete[] m_data;
}

template <typename KeyType, typename ValueType, class LeafType>  
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::nextIndex(KeyType x) const
{
	for(int i= numKeys() - 1;i >= 0;i--) {
        if(x >= m_data[i].key)
			//std::cout<<"route by key "<<m_data[i].key;
            return (static_cast<BNode *>(m_data[i].index));
    }
    //return (m_data[m_numKeys-1].index);
	return static_cast<BNode *>(firstIndex());;
}

template <typename KeyType, typename ValueType, class LeafType> 
const KeyType BNode<KeyType, ValueType, LeafType>::firstKey() const { return m_data[0].key; }

template <typename KeyType, typename ValueType, class LeafType> 
const KeyType BNode<KeyType, ValueType, LeafType>::lastKey() const { return m_data[numKeys() - 1].key; }

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::insert(Pair<KeyType, ValueType> x)
{
	if(isRoot()) 
		insertRoot(x);
	else if(isLeaf())
		insertLeaf(x);
	else
		insertInterior(x);
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::remove(Pair<KeyType, ValueType> x)
{
	if(isLeaf()) 
		removeLeaf(x);
	else {
		if(hasChildren()) {
			BNode * n = nextIndex(x.key);
			n->remove(x);
		}
		else {
		    std::cout<<*this<<"has no child";
			removeKey(x.key);
			setFirstIndex(NULL);
		}
	}
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::insertRoot(Pair<KeyType, ValueType> x)
{
	if(hasChildren()) {
		BNode * n = nextIndex(x.key);
		n->insert(x);
	}
	else {
		BNode * dst = this;
		if(!hasKey(x.key)) {
			if(isFull()) {
				dst = splitRoot(x.key);
			}
			else
				insertKey(x.key);
		}
		//std::cout<<"set value in "<<*dst;
		dst->setValue(x);
	}
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::setValue(const Pair<KeyType, ValueType> & x)
{
	int i = findKey(x.key);
	if(i<0) return;
	if(!m_data[i].index) {
		//std::cout<<"value index is null";
		m_data[i].index = new LeafType(this);
	}
	//std::cout<<" key is "<<x.key<<" list size is "<<static_cast<LeafType *>(m_data[i].index)->size()<<"\n";
	static_cast<LeafType *>(m_data[i].index)->insert(*x.index);
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::insertLeaf(Pair<KeyType, ValueType> x)
{
	BNode * dst = this;
	if(!hasKey(x.key)) {
		if(isFull())
			dst = splitLeaf(x);
		else
			insertKey(x.key);
	}
	dst->setValue(x);
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::insertInterior(Pair<KeyType, ValueType> x)
{
	BNode * n = nextIndex(x.key);
	n->insert(x);
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::insertData(Pair<KeyType, Entity> x)
{	
	int i;
    for(i= numKeys() - 1;i >= 0 && m_data[i].key > x.key; i--)
        m_data[i+1] = m_data[i];
		
    m_data[i+1] = x;
	//std::cout<<"insert key "<<x.key<<" at "<<i+1<<"\n";
    increaseNumKeys();
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::insertKey(KeyType x)
{	
	int i;
    for(i= numKeys() - 1;i >= 0 && m_data[i].key > x; i--)
        m_data[i+1] = m_data[i];
		
    m_data[i+1].key = x;
	m_data[i+1].index = NULL;
	//std::cout<<"insert key "<<x.key<<" at "<<i+1<<"\n";
    increaseNumKeys();
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::splitRoot(KeyType x)
{
	std::cout<<"split root "<<*this;

	BNode * one = new BNode(this); one->setLeaf();
	BNode * two = new BNode(this); two->setLeaf();
	
	Pair<KeyType, Entity> ex;
	ex.key = x;
	partData(ex, m_data, one, two, true);
	
	std::cout<<"into "<<*one<<*two;
	
	setFirstIndex(one);
	m_data[0].key = two->firstKey();
	m_data[0].index = two;
	one->connectSibling(two);
	setNumKeys(1);
	one->balanceLeafLeft();
	
	if(one->hasKey(x)) return one;
	return two;
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::splitLeaf(Pair<KeyType, ValueType> x)
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
	ex.key = x.key;
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
	
	if(two->hasKey(x.key)) return two;
	return this;
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::bounce(Pair<KeyType, Entity> b)
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

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const
{
	if(isLeaf()) return;
	if(!hasChildren()) return;
	dst[level].push_back(firstIndex());
	for(int i = 0;i < numKeys(); i++)
		dst[level].push_back(m_data[i].index);
		
	level++;
		
	BNode * n = static_cast<BNode *>(firstIndex());
	n->getChildren(dst, level);
	for(int i = 0;i < numKeys(); i++) {
		n = static_cast<BNode *>(m_data[i].index);
		n->getChildren(dst, level);
	}
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::firstLeaf()
{
	if(isRoot()) { 
		if(hasChildren())
			return static_cast<BNode *>(firstIndex())->firstLeaf();
		else 
			return this;
			
	}
	
	if(isLeaf())
		return this;
	
	return static_cast<BNode *>(firstIndex())->firstLeaf();;
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::connectChildren()
{
	if(!hasChildren()) return;
	Entity * n = firstIndex();
	n->setParent(this);
	for(int i = 0;i < numKeys(); i++) {
		n = m_data[i].index;
		n->setParent(this);
	}
}


template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::partRoot(Pair<KeyType, Entity> x)
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

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::partInterior(Pair<KeyType, Entity> x)
{
	std::cout<<"part interior "<<*this;
	
	BNode * rgt = new BNode(parent());
	
	Pair<KeyType, Entity> * old = new Pair<KeyType, Entity>[MaxNumKeysPerNode];
	for(int i=0; i < MaxNumKeysPerNode; i++)
		old[i] = m_data[i];
	
	setNumKeys(0);
	Pair<KeyType, Entity> p = partData(x, old, this, rgt);
	
	delete[] old;
	
	std::cout<<"into "<<*this<<*rgt;
	
	connectChildren();
	
	rgt->setFirstIndex(p.index);
	rgt->connectChildren();
	
	Pair<KeyType, Entity> b;
	b.key = p.key;
	b.index = rgt;
	parentNode()->bounce(b);
}

template <typename KeyType, typename ValueType, class LeafType> 
Pair<KeyType, Entity> BNode<KeyType, ValueType, LeafType>::partData(Pair<KeyType, Entity> x, Pair<KeyType, Entity> old[], BNode * lft, BNode * rgt, bool doSplitLeaf)
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

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::balanceLeaf()
{
	if(!balanceLeafRight()) {
		balanceLeafLeft();
	}
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::balanceLeafRight()
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

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::balanceLeafLeft()
{
	const Pair<KeyType, Entity> s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s.key, found);
	
	if(!found) return;
	
	BNode * leftSibling = crossed->leafLeftTo(s);
	
	if(leftSibling == this) return;
	
	int k = shouldBalance(leftSibling, this);
	if(k == 0) return;
	
	Pair<KeyType, Entity> old = firstData();
	if(k < 0) leftSibling->rightData(-k, this);
	else this->leftData(k, leftSibling);
	
	crossed->replaceKey(old.key, firstData().key);
	
	std::cout<<"\nbalanced "<<*leftSibling<<*this;
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::ancestor(const KeyType & x, bool & found) const
{
	if(parentNode()->hasKey(x)) {
		found = true;
		return parentNode();
	}
	
	if(parentNode()->isRoot()) return NULL;
	return parentNode()->ancestor(x, found);
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::hasKey(const KeyType & x) const
{
    if(x > lastKey() || x < firstKey()) return false;
	return findKey(x) > -1;
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::leftTo(const Pair<KeyType, Entity> & x) const
{
	if(numKeys()==1) return static_cast<BNode *>(firstIndex());
	int i;
	for(i = numKeys() - 1; i >= 0; i--) {		
        if(m_data[i].key < x.key) {
			return static_cast<BNode *>(m_data[i].index);
		}
    }
	return static_cast<BNode *>(firstIndex());
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::rightTo(const Pair<KeyType, Entity> & x, Pair<KeyType, Entity> & k) const
{
	int i;
	for(i = 0; i < numKeys(); i++) {		
        if(m_data[i].key >= x.key) {
			k = m_data[i];
			return static_cast<BNode *>(m_data[i].index);
		}
    }
	return NULL;
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::leafLeftTo(Pair<KeyType, Entity> x)
{
	if(isLeaf()) return this;
	
	BNode * n = leftTo(x);
	return n->leafLeftTo(x);
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::rightData(int num, BNode * rgt)
{
	for(int i = 0; i < num; i++) {
		rgt->insertData(lastData());
		removeLastData();
	}
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::leftData(int num, BNode * lft)
{
	for(int i = 0; i < num; i++) {
		lft->insertData(firstData());
		removeFirstData();
	}
}

template <typename KeyType, typename ValueType, class LeafType> 
Pair<KeyType, Entity> BNode<KeyType, ValueType, LeafType>::lastData() const { return m_data[numKeys() - 1]; }

template <typename KeyType, typename ValueType, class LeafType> 
Pair<KeyType, Entity> BNode<KeyType, ValueType, LeafType>::firstData() const { return m_data[0]; }

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::removeLastData()
{
	reduceNumKeys();
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::removeFirstData()
{
	for(int i = 0; i < numKeys() - 1; i++) {
		m_data[i] = m_data[i+1];
	}
	reduceNumKeys();
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::replaceKey(KeyType x, KeyType y)
{
	for(int i = 0; i < numKeys(); i++) {
		if(m_data[i].key == x) {
			m_data[i].key = y;
			return;
		}
	}
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::replaceIndex(int n, Pair<KeyType, Entity> x)
{
	m_data[n].index = x.index;
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::removeLeaf(const Pair<KeyType, ValueType> & x)
{
	if(!removeDataLeaf(x)) return;
	if(!underflow()) return;
	
	if(!mergeLeaf())
		balanceLeaf();
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::mergeLeaf()
{
	if(mergeLeafRight())
		return true;
		
	return mergeLeafLeft();
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::mergeLeafRight()
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
	
	Pair<KeyType, Entity> k = up->dataRightTo(old);
	
	if(parent() == up) {
		k.index = this;
		int ki = up->keyLeft(k.key);
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

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::mergeLeafLeft()
{
	const Pair<KeyType, Entity> s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s.key, found);
	
	if(!found) return false;
	
	BNode * leftSibling = crossed->leafLeftTo(s);
	
	if(leftSibling == this) return false;
	
	return leftSibling->mergeLeafRight();
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::removeDataLeaf(const Pair<KeyType, ValueType> & x)
{
	int i, found = -1;
    for(i= 0;i < numKeys(); i++) {
        if(m_data[i].key == x.key) {
			found = i;
			break;
		}
	}
	
	if(found < 0) return false;
	
	if(found == numKeys() - 1) {
		reduceNumKeys();
		return true;
	}
	
	for(i= found; i < numKeys() - 1; i++)
        m_data[i] = m_data[i+1];
		
	if(found == 0) {
		bool c = false;
		BNode * crossed = ancestor(x.key, c);
		if(c) crossed->replaceKey(x.key, firstData().key);
	}
		
    reduceNumKeys();
	return true;
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::removeKey(const KeyType & x)
{
	int i, found = -1;
    for(i= 0;i < numKeys(); i++) {
        if(m_data[i].key == x) {
			found = i;
			break;
		}
	}
	
	if(found < 0) return false;
	
	if(found == 0) 
		setFirstIndex(m_data[found].index);
	else
		m_data[found - 1].index = m_data[found].index;

	if(found == numKeys() - 1) {
		reduceNumKeys();
		return true;
	}
	
	for(i= found; i < numKeys() - 1; i++)
		m_data[i] = m_data[i+1];
		
    reduceNumKeys();
	return true;
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::pop(const Pair<KeyType, Entity> & x)
{
	if(isRoot()) popRoot(x);
	else popInterior(x);
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::popRoot(const Pair<KeyType, Entity> & x)
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
			
		delete lft;
		
		connectChildren();
	}
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::popInterior(const Pair<KeyType, Entity> & x)
{
	removeKey(x.key);
	if(!underflow()) return;
	std::cout<<"interior underflow! "<<*this;
	if(!mergeInterior())
		balanceInterior();
}

template <typename KeyType, typename ValueType, class LeafType> 
const Pair<KeyType, Entity> BNode<KeyType, ValueType, LeafType>::data(int x) const { return m_data[x]; }

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::mergeData(BNode * another, int start)
{
	const int num = another->numKeys();
	for(int i = start; i < num; i++)
		insertData(another->data(i));
}

template <typename KeyType, typename ValueType, class LeafType> 
const Pair<KeyType, Entity> BNode<KeyType, ValueType, LeafType>::dataRightTo(const Pair<KeyType, Entity> & x) const
{
	const int num = numKeys();
	for(int i = 0; i < num; i++) {
		if(m_data[i].key >= x.key) return m_data[i];
	}
	return lastData();
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::mergeInterior()
{
	if(mergeInteriorRight()) return true;
	return mergeInteriorLeft();
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::balanceInterior()
{
	if(!balanceInteriorRight()) balanceInteriorLeft();
}



template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::mergeInteriorRight()
{
	Pair<KeyType, Entity> k;
	BNode * rgt = parentNode()->rightTo(lastData(), k);
	if(!rgt) return false;
	if(!shouldInteriorMerge(this, rgt)) return false;
	
	k.index = rgt->firstIndex();
	
	insertData(k);
	
	rgt->leftData(rgt->numKeys(), this);
	
	delete rgt;
	
	connectChildren();
	
	int ki = parentNode()->keyLeft(k.key); 
	k.index = this;
	parentNode()->setData(ki, k);
	
	parentNode()->pop(k);
	return true;
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::mergeInteriorLeft()
{
	BNode * lft =leftInteriorNeighbor();
	if(!lft) return false;
	
	return lft->mergeInteriorRight();
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::leftInteriorNeighbor() const
{
	Pair<KeyType, Entity> k, j;
	if(!parentNode()->dataLeftTo(firstData(), k)) return NULL;
	j = k;
	if(j.index == this)
		parentNode()->dataLeftTo(k, j);
	
	if(j.index == this) return NULL;
	return static_cast<BNode *>(j.index);
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::setData(int k, const Pair<KeyType, Entity> & x)
{
	m_data[k] = x;
}

template <typename KeyType, typename ValueType, class LeafType> 
int BNode<KeyType, ValueType, LeafType>::keyLeft(const KeyType & x) const
{
	for(int i = 0; i < numKeys(); i++) {		
        if(m_data[i].key >= x) {
			return i;
		}
    }
	return -1;
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::dataLeftTo(const Pair<KeyType, Entity> & x, Pair<KeyType, Entity> & dst) const
{
	for(int i = numKeys() - 1; i >= 0; i--) {		
        if(m_data[i].key < x.key) {
			dst = m_data[i];
			return true;
		}
    }
	dst.index = firstIndex();
	return false;
}

template <typename KeyType, typename ValueType, class LeafType> 
bool BNode<KeyType, ValueType, LeafType>::balanceInteriorRight()
{
	Pair<KeyType, Entity> k;
	BNode * rgt = parentNode()->rightTo(lastData(), k);
	if(!rgt) return false;

	k.index = rgt->firstIndex();
	
	insertData(k);
	
	parentNode()->replaceKey(k.key, rgt->firstData().key);
	
	rgt->removeKey(rgt->firstData().key);

	return true;
}

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::balanceInteriorLeft()
{
	BNode * lft = leftInteriorNeighbor();
	if(!lft) return;
	Pair<KeyType, Entity> k;
	parentNode()->rightTo(lft->lastData(), k);
	k.index = firstIndex();
	insertData(k);
	Pair<KeyType, Entity> l = lft->lastData();
	setFirstIndex(l.index);
	parentNode()->replaceKey(k.key, l.key);
	lft->removeLastData();
}

template <typename KeyType, typename ValueType, class LeafType> 
BNode<KeyType, ValueType, LeafType> * BNode<KeyType, ValueType, LeafType>::parentNode() const
{
	return static_cast<BNode *>(parent());
}

template <typename KeyType, typename ValueType, class LeafType> 
const std::string BNode<KeyType, ValueType, LeafType>::str() const 
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

template <typename KeyType, typename ValueType, class LeafType> 
void BNode<KeyType, ValueType, LeafType>::getValues(std::vector<ValueType> & dst)
{
	for(int i = 0; i < numKeys(); i++) {
		static_cast<LeafType *>(m_data[i].index)->getValues(dst);
	}
}

template <typename KeyType, typename ValueType, class LeafType> 
int BNode<KeyType, ValueType, LeafType>::findKey(const KeyType & x) const
{
    int found = -1;
    int lo = 0, hi = numKeys() - 1;
    int mid;
        
    while(lo <= hi) {
        mid = (lo + hi) / 2;
        
        if(key(lo) == x) found = lo;
        else if(key(hi) == x) found = hi;
        else if(key(mid) == x) found = mid;
        
        if(found > -1) break;
        
        if(x < key(mid)) hi = mid;
        else lo = mid;
        
        if(lo >= hi - 1) break;
    }
    
    return found;
}

} // end of namespace sdb
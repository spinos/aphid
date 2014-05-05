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
#include <Pair.h>

namespace sdb {
#define MAXPERNODEKEYCOUNT 4
#define MINPERNODEKEYCOUNT 2
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
	
	virtual void display() const;
private:
	Entity *m_first;
	bool m_isLeaf;
};

template <typename KeyType>
class BNode : public TreeNode
{
public:
	BNode(Entity * parent = NULL);
	virtual ~BNode() {}
	
	void insert(Pair<KeyType, Entity> x);
    void remove(Pair<KeyType, Entity> x);
	
    void getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const;
	
	virtual void display() const;
	
private:
	KeyType firstKey() const;
    BNode *nextIndex(int x) const;
	
	void insertRoot(Pair<KeyType, Entity> x);
	void splitRoot(Pair<KeyType, Entity> x);
	
	void insertLeaf(Pair<KeyType, Entity> x);
	void splitLeaf(Pair<KeyType, Entity> x);
	
	void insertData(Pair<KeyType, Entity> x);
	
	void partRoot(Pair<KeyType, Entity> x);
	Pair<KeyType, Entity> partData(Pair<KeyType, Entity> x, Pair<KeyType, Entity> old[], BNode * lft, BNode * rgt, bool doSplitLeaf = false);
	
	void partInterior(Pair<KeyType, Entity> x);
	
	void insertInterior(Pair<KeyType, Entity> x);
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

	void replaceKey(Pair<KeyType, Entity> x, Pair<KeyType, Entity> y);
	
	BNode * ancestor(const Pair<KeyType, Entity> & x, bool & found) const;
	BNode * leftTo(const Pair<KeyType, Entity> & x) const;
	BNode * rightTo(const Pair<KeyType, Entity> & x, Pair<KeyType, Entity> & k) const;
	BNode * leafLeftTo(Pair<KeyType, Entity> x);
	
	bool hasKey(Pair<KeyType, Entity> x) const;
	
	void removeLeaf(const Pair<KeyType, Entity> & x);
	bool removeData(const Pair<KeyType, Entity> & x);
	bool removeDataLeaf(const Pair<KeyType, Entity> & x);

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
	int dataId(const Pair<KeyType, Entity> & x) const;
	
	bool dataLeftTo(const Pair<KeyType, Entity> & x, Pair<KeyType, Entity> & dst) const;
	
	BNode * leftInteriorNeighbor() const;
	
	bool balanceInteriorRight();
	void balanceInteriorLeft();
	
	BNode * parentNode() const;
	
	bool isFull() const { return m_numKeys == MAXPERNODEKEYCOUNT; }
	bool underflow() const { return m_numKeys < MINPERNODEKEYCOUNT; }
	int numKeys() const  { return m_numKeys; }
	
	void reduceNumKeys() { m_numKeys--; }
	void increaseNumKeys() { m_numKeys++; }
	void setNumKeys(int x) { m_numKeys = x; }
	
	bool shouldMerge(BNode * lft, BNode * rgt) const { return (lft->numKeys() + rgt->numKeys()) <= MAXPERNODEKEYCOUNT; }
	bool shouldInteriorMerge(BNode * lft, BNode * rgt) const { return (lft->numKeys() + rgt->numKeys()) < MAXPERNODEKEYCOUNT; }
	int shouldBalance(BNode * lft, BNode * rgt) const { return (lft->numKeys() + rgt->numKeys()) / 2 - lft->numKeys(); }
	
private:
    Pair<KeyType, Entity> m_data[MAXPERNODEKEYCOUNT];
	int m_numKeys;
};

template <typename KeyType>  
BNode<KeyType>::BNode(Entity * parent) : TreeNode(parent)
{
	m_numKeys = 0;
	for(int i=0;i< MAXPERNODEKEYCOUNT;i++)
        m_data[i].index = NULL;
}

template <typename KeyType>  
BNode<KeyType> * BNode<KeyType>::nextIndex(int x) const
{
	for(int i= numKeys() - 1;i >= 0;i--) {
        if(x >= m_data[i].key)
			//std::cout<<"route by key "<<m_data[i].key;
            return (static_cast<BNode *>(m_data[i].index));
    }
    //return (m_data[m_numKeys-1].index);
	return static_cast<BNode *>(firstIndex());;
}

template <typename KeyType> 
void BNode<KeyType>::display() const
{
    int i;
    std::cout<<"(";
    for(i=0;i< numKeys();i++) {
		std::cout<<m_data[i].key;
		if(i< numKeys()-1) std::cout<<" ";
	}
    std::cout<<") ";
}

template <typename KeyType> 
KeyType BNode<KeyType>::firstKey() const { return m_data[0].key; }

template <typename KeyType> 
void BNode<KeyType>::insert(Pair<KeyType, Entity> x)
{
	if(isRoot()) 
		insertRoot(x);
	else if(isLeaf())
		insertLeaf(x);
	else
		insertInterior(x);
}

template <typename KeyType> 
void BNode<KeyType>::insertRoot(Pair<KeyType, Entity> x)
{
	if(hasChildren()) {
		BNode * n = nextIndex(x.key);
		n->insert(x);
	}
	else {
		if(isFull()) {
			splitRoot(x);
		}
		else
			insertData(x);
	}
}

template <typename KeyType> 
void BNode<KeyType>::insertLeaf(Pair<KeyType, Entity> x)
{
	if(isFull()) {
		splitLeaf(x);
	}
	else {
		insertData(x);
	}
}

template <typename KeyType> 
void BNode<KeyType>::insertInterior(Pair<KeyType, Entity> x)
{
	BNode * n = nextIndex(x.key);
	n->insert(x);
}

template <typename KeyType> 
void BNode<KeyType>::insertData(Pair<KeyType, Entity> x)
{	
	int i;
    for(i= numKeys() - 1;i >= 0 && m_data[i].key > x.key; i--)
        m_data[i+1] = m_data[i];
		
    m_data[i+1]= x;
	//std::cout<<"insert key "<<x.key<<" at "<<i+1<<"\n";
    increaseNumKeys();
}

template <typename KeyType> 
void BNode<KeyType>::splitRoot(Pair<KeyType, Entity> x)
{
	std::cout<<"split root ";
	display();
	BNode * one = new BNode(this); one->setLeaf();
	BNode * two = new BNode(this); two->setLeaf();
	
	partData(x, m_data, one, two, true);
	
	std::cout<<"into ";
	one->display();
	two->display();
	
	setFirstIndex(one);
	m_data[0].key = two->firstKey();
	m_data[0].index = two;
	one->connectSibling(two);
	setNumKeys(1);
	one->balanceLeafLeft();
}

template <typename KeyType> 
void BNode<KeyType>::splitLeaf(Pair<KeyType, Entity> x)
{
	//std::cout<<"split leaf ";
	//display();
	//std::cout<<"into ";
	Entity * oldRgt = sibling();
	BNode * two = new BNode(parent()); two->setLeaf();

	Pair<KeyType, Entity> old[MAXPERNODEKEYCOUNT];
	for(int i=0; i < MAXPERNODEKEYCOUNT; i++)
		old[i] = m_data[i];
		
	setNumKeys(0);
	partData(x, old, this, two, true);
	//display();
	//two->display();
	
	connectSibling(two);
	if(oldRgt) two->connectSibling(oldRgt);
	
	Pair<KeyType, Entity> b;
	b.key = two->firstKey();
	b.index = two;
	parentNode()->bounce(b);
	//balanceLeafLeft();
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
		dst[level].push_back(m_data[i].index);
		
	level++;
		
	BNode * n = static_cast<BNode *>(firstIndex());
	n->getChildren(dst, level);
	for(int i = 0;i < numKeys(); i++) {
		n = static_cast<BNode *>(m_data[i].index);
		n->getChildren(dst, level);
	}
}

template <typename KeyType> 
void BNode<KeyType>::connectChildren()
{
	if(!hasChildren()) return;
	Entity * n = firstIndex();
	n->setParent(this);
	for(int i = 0;i < numKeys(); i++) {
		n = m_data[i].index;
		n->setParent(this);
	}
}


template <typename KeyType> 
void BNode<KeyType>::partRoot(Pair<KeyType, Entity> x)
{
	//std::cout<<"part root ";
	//display();
	BNode * one = new BNode(this);
	BNode * two = new BNode(this);
	
	Pair<KeyType, Entity> p = partData(x, m_data, one, two);
	
	//std::cout<<"into ";
	//one->display();
	//two->display();
	
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
	std::cout<<"part interior ";
	display();
	BNode * rgt = new BNode(parent());
	
	Pair<KeyType, Entity> old[MAXPERNODEKEYCOUNT];
	for(int i=0; i < MAXPERNODEKEYCOUNT; i++)
		old[i] = m_data[i];
	
	setNumKeys(0);
	Pair<KeyType, Entity> p = partData(x, old, this, rgt);
	
	std::cout<<"into ";
	display();
	rgt->display();
	
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
	for(int i = MAXPERNODEKEYCOUNT - 1;i >= 0; i--) {
		if(x.key > old[i].key && !inserted) {
			q = x;
			i++;
			inserted = true;
		}
		else
			q = old[i];
			
		numKeysRight++;
		
		if(numKeysRight == MAXPERNODEKEYCOUNT / 2 + 1) {
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
	BNode * crossed = ancestor(s, found);
	
	if(!found) return false;
	
	int k = shouldBalance(this, rgtNode);
	if(k == 0) return false;
	
	Pair<KeyType, Entity> old = rgtNode->firstData();
	if(k < 0) rightData(-k, rgtNode);
	else rgtNode->leftData(k, this);
	
	crossed->replaceKey(old, rgtNode->firstData());
	
	return true;
}

template <typename KeyType> 
void BNode<KeyType>::balanceLeafLeft()
{
	const Pair<KeyType, Entity> s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s, found);
	
	if(!found) return;
	
	BNode * leftSibling = crossed->leafLeftTo(s);
	
	if(leftSibling == this) return;
	
	int k = shouldBalance(leftSibling, this);
	if(k == 0) return;
	
	Pair<KeyType, Entity> old = firstData();
	if(k < 0) leftSibling->rightData(-k, this);
	else this->leftData(k, leftSibling);
	
	crossed->replaceKey(old, firstData());
	
	std::cout<<"\nbalanced ";
	leftSibling->display();
	display();
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::ancestor(const Pair<KeyType, Entity> & x, bool & found) const
{
	if(parentNode()->hasKey(x)) {
		found = true;
		return parentNode();
	}
	
	if(parentNode()->isRoot()) return NULL;
	return parentNode()->ancestor(x, found);
}

template <typename KeyType> 
bool BNode<KeyType>::hasKey(Pair<KeyType, Entity> x) const
{
	for(int i = 0; i < numKeys(); i++) {
		if(m_data[i].key == x.key)
			return true;
	}
	return false;
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::leftTo(const Pair<KeyType, Entity> & x) const
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

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::rightTo(const Pair<KeyType, Entity> & x, Pair<KeyType, Entity> & k) const
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

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::leafLeftTo(Pair<KeyType, Entity> x)
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
	reduceNumKeys();
}

template <typename KeyType> 
void BNode<KeyType>::removeFirstData()
{
	for(int i = 0; i < numKeys() - 1; i++) {
		m_data[i] = m_data[i+1];
	}
	reduceNumKeys();
}

template <typename KeyType> 
void BNode<KeyType>::replaceKey(Pair<KeyType, Entity> x, Pair<KeyType, Entity> y)
{
	for(int i = 0; i < numKeys(); i++) {
		if(m_data[i].key == x.key) {
			m_data[i].key = y.key;
			return;
		}
	}
}

template <typename KeyType> 
void BNode<KeyType>::replaceIndex(int n, Pair<KeyType, Entity> x)
{
	m_data[n].index = x.index;
}

template <typename KeyType> 
void BNode<KeyType>::remove(Pair<KeyType, Entity> x)
{
	if(isLeaf()) 
		removeLeaf(x);
	else {
		if(hasChildren()) {
			BNode * n = nextIndex(x.key);
			n->remove(x);
		}
		else
			removeData(x);
	}
}

template <typename KeyType> 
void BNode<KeyType>::removeLeaf(const Pair<KeyType, Entity> & x)
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
	
	Pair<KeyType, Entity> k = up->dataRightTo(old);
	
	if(parent() == up) {
		k.index = this;
		int ki = up->dataId(k);
		up->setData(ki, k);
	}
	else {
		bool found = false;
		BNode * crossed = ancestor(s, found);
		crossed->replaceKey(s, k);
	}
	
	up->pop(k);
	
	return true;
}

template <typename KeyType> 
bool BNode<KeyType>::mergeLeafLeft()
{
	const Pair<KeyType, Entity> s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s, found);
	
	if(!found) return false;
	
	BNode * leftSibling = crossed->leafLeftTo(s);
	
	if(leftSibling == this) return false;
	
	return leftSibling->mergeLeafRight();
}

template <typename KeyType> 
bool BNode<KeyType>::removeDataLeaf(const Pair<KeyType, Entity> & x)
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
		BNode * crossed = ancestor(x, c);
		if(c) crossed->replaceKey(x, firstData());
	}
		
    reduceNumKeys();
	return true;
}

template <typename KeyType> 
bool BNode<KeyType>::removeData(const Pair<KeyType, Entity> & x)
{
	int i, found = -1;
    for(i= 0;i < numKeys(); i++) {
        if(m_data[i].key == x.key) {
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

template <typename KeyType> 
void BNode<KeyType>::pop(const Pair<KeyType, Entity> & x)
{
	if(isRoot()) popRoot(x);
	else popInterior(x);
}

template <typename KeyType> 
void BNode<KeyType>::popRoot(const Pair<KeyType, Entity> & x)
{
	if(numKeys() > 1) removeData(x);
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

template <typename KeyType> 
void BNode<KeyType>::popInterior(const Pair<KeyType, Entity> & x)
{
	removeData(x);
	if(!underflow()) return;
	std::cout<<"interior underflow! ";display();
	if(!mergeInterior())
		balanceInterior();
}

template <typename KeyType> 
const Pair<KeyType, Entity> BNode<KeyType>::data(int x) const { return m_data[x]; }

template <typename KeyType> 
void BNode<KeyType>::mergeData(BNode * another, int start)
{
	const int num = another->numKeys();
	for(int i = start; i < num; i++)
		insertData(another->data(i));
}

template <typename KeyType> 
const Pair<KeyType, Entity> BNode<KeyType>::dataRightTo(const Pair<KeyType, Entity> & x) const
{
	const int num = numKeys();
	for(int i = 0; i < num; i++) {
		if(m_data[i].key >= x.key) return m_data[i];
	}
	return lastData();
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
	BNode * rgt = parentNode()->rightTo(lastData(), k);
	if(!rgt) return false;
	if(!shouldInteriorMerge(this, rgt)) return false;
	
	k.index = rgt->firstIndex();
	
	insertData(k);
	
	rgt->leftData(rgt->numKeys(), this);
	
	delete rgt;
	
	connectChildren();
	
	int ki = parentNode()->dataId(k); 
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
	if(!parentNode()->dataLeftTo(firstData(), k)) return NULL;
	j = k;
	if(j.index == this)
		parentNode()->dataLeftTo(k, j);
	
	if(j.index == this) return NULL;
	return static_cast<BNode *>(j.index);
}

template <typename KeyType> 
void BNode<KeyType>::setData(int k, const Pair<KeyType, Entity> & x)
{
	m_data[k] = x;
}

template <typename KeyType> 
int BNode<KeyType>::dataId(const Pair<KeyType, Entity> & x) const
{
	for(int i = 0; i < numKeys(); i++) {		
        if(m_data[i].key >= x.key) {
			return i;
		}
    }
	return -1;
}

template <typename KeyType> 
bool BNode<KeyType>::dataLeftTo(const Pair<KeyType, Entity> & x, Pair<KeyType, Entity> & dst) const
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

template <typename KeyType> 
bool BNode<KeyType>::balanceInteriorRight()
{
	Pair<KeyType, Entity> k;
	BNode * rgt = parentNode()->rightTo(lastData(), k);
	if(!rgt) return false;

	k.index = rgt->firstIndex();
	
	insertData(k);
	
	parentNode()->replaceKey(k, rgt->firstData());
	
	rgt->removeData(rgt->firstData());

	return true;
}

template <typename KeyType> 
void BNode<KeyType>::balanceInteriorLeft()
{
	BNode * lft = leftInteriorNeighbor();
	if(!lft) return;
	Pair<KeyType, Entity> k;
	parentNode()->rightTo(lft->lastData(), k);
	k.index = firstIndex();
	insertData(k);
	Pair<KeyType, Entity> l = lft->lastData();
	setFirstIndex(l.index);
	parentNode()->replaceKey(k, l);
	lft->removeLastData();
}

template <typename KeyType> 
BNode<KeyType> * BNode<KeyType>::parentNode() const
{
	return static_cast<BNode *>(parent());
}
} // end of namespace sdb
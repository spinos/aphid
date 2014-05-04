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

#include <Pair.h>

namespace sdb {

template <typename KeyType, class IndexType>  
class BNode : public Entity
{
public:
	BNode(Entity * parent = NULL);
	virtual ~BNode() {}
	
	void insert(Pair<KeyType, IndexType> x);
    void remove(Pair<KeyType, IndexType> x);
	
	int firstKey() const;
    BNode *nextIndex(int x) const;
	
    virtual void display() const;
	void getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const;
	
private:
	void insertRoot(Pair<KeyType, IndexType> x);
	void splitRoot(Pair<KeyType, IndexType> x);
	
	void insertLeaf(Pair<KeyType, IndexType> x);
	void splitLeaf(Pair<KeyType, IndexType> x);
	
	void insertData(Pair<KeyType, IndexType> x);
	
	void partRoot(Pair<KeyType, IndexType> x);
	Pair<KeyType, IndexType> partData(Pair<KeyType, IndexType> x, Pair<KeyType, IndexType> old[], BNode * lft, BNode * rgt, bool doSplitLeaf = false);
	
	void partInterior(Pair<KeyType, IndexType> x);
	
	void insertInterior(Pair<KeyType, IndexType> x);
	void connectChildren();
	
	void bounce(Pair<KeyType, IndexType> b);

	void balanceLeaf();
	bool balanceLeafRight();
	void balanceLeafLeft();
	void rightData(int num, BNode * rgt);
	void leftData(int num, BNode * lft);
	
	Pair<KeyType, IndexType> lastData() const;
	Pair<KeyType, IndexType> firstData() const;
	
	void removeLastData();
	void removeFirstData();

	void replaceKey(Pair<KeyType, IndexType> x, Pair<KeyType, IndexType> y);
	
	BNode * ancestor(const Pair<KeyType, IndexType> & x, bool & found) const;
	BNode * leftTo(const Pair<KeyType, IndexType> & x) const;
	BNode * rightTo(const Pair<KeyType, IndexType> & x, Pair<KeyType, IndexType> & k) const;
	BNode * leafLeftTo(Pair<KeyType, IndexType> x);
	
	bool hasKey(Pair<KeyType, IndexType> x) const;
	
	void removeLeaf(const Pair<KeyType, IndexType> & x);
	bool removeData(const Pair<KeyType, IndexType> & x);
	bool removeDataLeaf(const Pair<KeyType, IndexType> & x);

	bool mergeLeaf();
	bool mergeLeafRight();
	bool mergeLeafLeft();
	
	void pop(const Pair<KeyType, IndexType> & x);
	void popRoot(const Pair<KeyType, IndexType> & x);
	void popInterior(const Pair<KeyType, IndexType> & x);
	
	const Pair<KeyType, IndexType> data(int x) const;
	void mergeData(BNode * another, int start = 0);
	void replaceIndex(int n, Pair<KeyType, IndexType> x);
	
	const Pair<KeyType, IndexType> dataRightTo(const Pair<KeyType, IndexType> & x) const;
	
	bool mergeInterior();
	void balanceInterior();
	
	bool mergeInteriorRight();
	bool mergeInteriorLeft();
	
	void setData(int k, const Pair<KeyType, IndexType> & x);
	int dataId(const Pair<KeyType, IndexType> & x) const;
	
	bool dataLeftTo(const Pair<KeyType, IndexType> & x, Pair<KeyType, IndexType> & dst) const;
	
	BNode * leftInteriorNeighbor() const;
	
	bool balanceInteriorRight();
	void balanceInteriorLeft();
	
	BNode * parentNode() const;
	
private:
    Pair<KeyType, IndexType> m_data[MAXPERNODEKEYCOUNT];
};

template <typename KeyType, class IndexType>  
BNode<KeyType, IndexType>::BNode(Entity * parent) : Entity(parent)
{
	for(int i=0;i< MAXPERNODEKEYCOUNT;i++)
        m_data[i].index = NULL;
}

template <typename KeyType, class IndexType>  
BNode<KeyType, IndexType> * BNode<KeyType, IndexType>::nextIndex(int x) const
{
	for(int i= numKeys() - 1;i >= 0;i--) {
        if(x >= m_data[i].key)
			//std::cout<<"route by key "<<m_data[i].key;
            return (static_cast<BNode *>(m_data[i].index));
    }
    //return (m_data[m_numKeys-1].index);
	return static_cast<BNode *>(firstIndex());;
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::display() const
{
    int i;
    std::cout<<"(";
    for(i=0;i< numKeys();i++) {
		std::cout<<m_data[i].key;
		if(i< numKeys()-1) std::cout<<" ";
	}
    std::cout<<") ";
}

template <typename KeyType, class IndexType> 
int BNode<KeyType, IndexType>::firstKey() const { return m_data[0].key; }

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::insert(Pair<KeyType, IndexType> x)
{
	if(isRoot()) 
		insertRoot(x);
	else if(isLeaf())
		insertLeaf(x);
	else
		insertInterior(x);
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::insertRoot(Pair<KeyType, IndexType> x)
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

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::insertLeaf(Pair<KeyType, IndexType> x)
{
	if(isFull()) {
		splitLeaf(x);
	}
	else {
		insertData(x);
		//balanceLeaf();
	}
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::insertInterior(Pair<KeyType, IndexType> x)
{
	BNode * n = nextIndex(x.key);
	n->insert(x);
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::insertData(Pair<KeyType, IndexType> x)
{	
	int i;
    for(i= numKeys() - 1;i >= 0 && m_data[i].key > x.key; i--)
        m_data[i+1] = m_data[i];
		
    m_data[i+1]= x;
	//std::cout<<"insert key "<<x.key<<" at "<<i+1<<"\n";
    increaseNumKeys();
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::splitRoot(Pair<KeyType, IndexType> x)
{
	//std::cout<<"split root ";
	//display();
	BNode * one = new BNode(this); one->setLeaf();
	BNode * two = new BNode(this); two->setLeaf();
	
	partData(x, m_data, one, two, true);
	
	//std::cout<<"into ";
	//one->display();
	//two->display();
	
	setFirstIndex(one);
	m_data[0].key = two->firstKey();
	m_data[0].index = two;
	one->connectSibling(two);
	setNumKeys(1);
	one->balanceLeafLeft();
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::splitLeaf(Pair<KeyType, IndexType> x)
{
	//std::cout<<"split leaf ";
	//display();
	//std::cout<<"into ";
	Entity * oldRgt = sibling();
	BNode * two = new BNode(parent()); two->setLeaf();

	Pair<KeyType, IndexType> old[MAXPERNODEKEYCOUNT];
	for(int i=0; i < MAXPERNODEKEYCOUNT; i++)
		old[i] = m_data[i];
		
	setNumKeys(0);
	partData(x, old, this, two, true);
	//display();
	//two->display();
	
	connectSibling(two);
	if(oldRgt) two->connectSibling(oldRgt);
	
	Pair<KeyType, IndexType> b;
	b.key = two->firstKey();
	b.index = two;
	parentNode()->bounce(b);
	//balanceLeafLeft();
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::bounce(Pair<KeyType, IndexType> b)
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

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const
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

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::connectChildren()
{
	if(!hasChildren()) return;
	Entity * n = firstIndex();
	n->setParent(this);
	for(int i = 0;i < numKeys(); i++) {
		n = m_data[i].index;
		n->setParent(this);
	}
}


template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::partRoot(Pair<KeyType, IndexType> x)
{
	//std::cout<<"part root ";
	//display();
	BNode * one = new BNode(this);
	BNode * two = new BNode(this);
	
	Pair<KeyType, IndexType> p = partData(x, m_data, one, two);
	
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

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::partInterior(Pair<KeyType, IndexType> x)
{
	std::cout<<"part interior ";
	display();
	BNode * rgt = new BNode(parent());
	
	Pair<KeyType, IndexType> old[MAXPERNODEKEYCOUNT];
	for(int i=0; i < MAXPERNODEKEYCOUNT; i++)
		old[i] = m_data[i];
	
	setNumKeys(0);
	Pair<KeyType, IndexType> p = partData(x, old, this, rgt);
	
	std::cout<<"into ";
	display();
	rgt->display();
	
	connectChildren();
	
	rgt->setFirstIndex(p.index);
	rgt->connectChildren();
	
	Pair<KeyType, IndexType> b;
	b.key = p.key;
	b.index = rgt;
	parentNode()->bounce(b);
}

template <typename KeyType, class IndexType> 
Pair<KeyType, IndexType> BNode<KeyType, IndexType>::partData(Pair<KeyType, IndexType> x, Pair<KeyType, IndexType> old[], BNode * lft, BNode * rgt, bool doSplitLeaf)
{
	Pair<KeyType, IndexType> res, q;
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

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::balanceLeaf()
{
	if(!balanceLeafRight()) {
		balanceLeafLeft();
	}
}

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::balanceLeafRight()
{
	Entity * rgt = sibling();
	if(!rgt) return false;
	
	BNode * rgtNode = static_cast<BNode *>(rgt);
	const Pair<KeyType, IndexType> s = rgtNode->firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s, found);
	
	if(!found) return false;
	
	int k = shouldBalance(this, rgt);
	if(k == 0) return false;
	
	Pair<KeyType, IndexType> old = rgtNode->firstData();
	if(k < 0) rightData(-k, rgtNode);
	else rgtNode->leftData(k, this);
	
	crossed->replaceKey(old, rgtNode->firstData());
	
	return true;
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::balanceLeafLeft()
{
	const Pair<KeyType, IndexType> s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s, found);
	
	if(!found) return;
	
	BNode * leftSibling = crossed->leafLeftTo(s);
	
	if(leftSibling == this) return;
	
	int k = shouldBalance(leftSibling, this);
	if(k == 0) return;
	
	Pair<KeyType, IndexType> old = firstData();
	if(k < 0) leftSibling->rightData(-k, this);
	else this->leftData(k, leftSibling);
	
	crossed->replaceKey(old, firstData());
	
	std::cout<<"\nbalanced ";
	leftSibling->display();
	display();
}

template <typename KeyType, class IndexType> 
BNode<KeyType, IndexType> * BNode<KeyType, IndexType>::ancestor(const Pair<KeyType, IndexType> & x, bool & found) const
{
	if(parentNode()->hasKey(x)) {
		found = true;
		return parentNode();
	}
	
	if(parent()->isRoot()) return NULL;
	return parentNode()->ancestor(x, found);
}

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::hasKey(Pair<KeyType, IndexType> x) const
{
	for(int i = 0; i < numKeys(); i++) {
		if(m_data[i].key == x.key)
			return true;
	}
	return false;
}

template <typename KeyType, class IndexType> 
BNode<KeyType, IndexType> * BNode<KeyType, IndexType>::leftTo(const Pair<KeyType, IndexType> & x) const
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

template <typename KeyType, class IndexType> 
BNode<KeyType, IndexType> * BNode<KeyType, IndexType>::rightTo(const Pair<KeyType, IndexType> & x, Pair<KeyType, IndexType> & k) const
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

template <typename KeyType, class IndexType> 
BNode<KeyType, IndexType> * BNode<KeyType, IndexType>::leafLeftTo(Pair<KeyType, IndexType> x)
{
	if(isLeaf()) return this;
	
	BNode * n = leftTo(x);
	return n->leafLeftTo(x);
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::rightData(int num, BNode * rgt)
{
	for(int i = 0; i < num; i++) {
		rgt->insertData(lastData());
		removeLastData();
	}
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::leftData(int num, BNode * lft)
{
	for(int i = 0; i < num; i++) {
		lft->insertData(firstData());
		removeFirstData();
	}
}

template <typename KeyType, class IndexType> 
Pair<KeyType, IndexType> BNode<KeyType, IndexType>::lastData() const { return m_data[numKeys() - 1]; }

template <typename KeyType, class IndexType> 
Pair<KeyType, IndexType> BNode<KeyType, IndexType>::firstData() const { return m_data[0]; }

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::removeLastData()
{
	reduceNumKeys();
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::removeFirstData()
{
	for(int i = 0; i < numKeys() - 1; i++) {
		m_data[i] = m_data[i+1];
	}
	reduceNumKeys();
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::replaceKey(Pair<KeyType, IndexType> x, Pair<KeyType, IndexType> y)
{
	for(int i = 0; i < numKeys(); i++) {
		if(m_data[i].key == x.key) {
			m_data[i].key = y.key;
			return;
		}
	}
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::replaceIndex(int n, Pair<KeyType, IndexType> x)
{
	m_data[n].index = x.index;
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::remove(Pair<KeyType, IndexType> x)
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

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::removeLeaf(const Pair<KeyType, IndexType> & x)
{
	if(!removeDataLeaf(x)) return;
	if(!underflow()) return;
	
	if(!mergeLeaf())
		balanceLeaf();
}

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::mergeLeaf()
{
	if(mergeLeafRight())
		return true;
		
	return mergeLeafLeft();
}

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::mergeLeafRight()
{
	Entity * rgt = sibling();
	if(!rgt) return false;
	
	BNode * rgtNode = static_cast<BNode *>(rgt);
	
	const Pair<KeyType, IndexType> s = rgtNode->firstData();
	
	if(!shouldMerge(this, rgt)) return false;
	
	BNode * up = rgtNode->parentNode();

	Pair<KeyType, IndexType> old = rgtNode->firstData();
	Entity * oldSibling = rgt->sibling();
	
	rgtNode->leftData(rgt->numKeys(), this);
	
	delete rgt;
	
	connectSibling(oldSibling);
	
	Pair<KeyType, IndexType> k = up->dataRightTo(old);
	
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

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::mergeLeafLeft()
{
	const Pair<KeyType, IndexType> s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s, found);
	
	if(!found) return false;
	
	BNode * leftSibling = crossed->leafLeftTo(s);
	
	if(leftSibling == this) return false;
	
	return leftSibling->mergeLeafRight();
}

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::removeDataLeaf(const Pair<KeyType, IndexType> & x)
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

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::removeData(const Pair<KeyType, IndexType> & x)
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

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::pop(const Pair<KeyType, IndexType> & x)
{
	if(isRoot()) popRoot(x);
	else popInterior(x);
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::popRoot(const Pair<KeyType, IndexType> & x)
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

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::popInterior(const Pair<KeyType, IndexType> & x)
{
	removeData(x);
	if(!underflow()) return;
	std::cout<<"interior underflow! ";display();
	if(!mergeInterior())
		balanceInterior();
}

template <typename KeyType, class IndexType> 
const Pair<KeyType, IndexType> BNode<KeyType, IndexType>::data(int x) const { return m_data[x]; }

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::mergeData(BNode * another, int start)
{
	const int num = another->numKeys();
	for(int i = start; i < num; i++)
		insertData(another->data(i));
}

template <typename KeyType, class IndexType> 
const Pair<KeyType, IndexType> BNode<KeyType, IndexType>::dataRightTo(const Pair<KeyType, IndexType> & x) const
{
	const int num = numKeys();
	for(int i = 0; i < num; i++) {
		if(m_data[i].key >= x.key) return m_data[i];
	}
	return lastData();
}

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::mergeInterior()
{
	if(mergeInteriorRight()) return true;
	return mergeInteriorLeft();
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::balanceInterior()
{
	if(!balanceInteriorRight()) balanceInteriorLeft();
}



template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::mergeInteriorRight()
{
	Pair<KeyType, IndexType> k;
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

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::mergeInteriorLeft()
{
	BNode * lft =leftInteriorNeighbor();
	if(!lft) return false;
	
	return lft->mergeInteriorRight();
}

template <typename KeyType, class IndexType> 
BNode<KeyType, IndexType> * BNode<KeyType, IndexType>::leftInteriorNeighbor() const
{
	Pair<KeyType, IndexType> k, j;
	if(!parentNode()->dataLeftTo(firstData(), k)) return NULL;
	j = k;
	if(j.index == this)
		parentNode()->dataLeftTo(k, j);
	
	if(j.index == this) return NULL;
	return static_cast<BNode *>(j.index);
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::setData(int k, const Pair<KeyType, IndexType> & x)
{
	m_data[k] = x;
}

template <typename KeyType, class IndexType> 
int BNode<KeyType, IndexType>::dataId(const Pair<KeyType, IndexType> & x) const
{
	for(int i = 0; i < numKeys(); i++) {		
        if(m_data[i].key >= x.key) {
			return i;
		}
    }
	return -1;
}

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::dataLeftTo(const Pair<KeyType, IndexType> & x, Pair<KeyType, IndexType> & dst) const
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

template <typename KeyType, class IndexType> 
bool BNode<KeyType, IndexType>::balanceInteriorRight()
{
	Pair<KeyType, IndexType> k;
	BNode * rgt = parentNode()->rightTo(lastData(), k);
	if(!rgt) return false;

	k.index = rgt->firstIndex();
	
	insertData(k);
	
	parentNode()->replaceKey(k, rgt->firstData());
	
	rgt->removeData(rgt->firstData());

	return true;
}

template <typename KeyType, class IndexType> 
void BNode<KeyType, IndexType>::balanceInteriorLeft()
{
	BNode * lft = leftInteriorNeighbor();
	if(!lft) return;
	Pair<KeyType, IndexType> k;
	parentNode()->rightTo(lft->lastData(), k);
	k.index = firstIndex();
	insertData(k);
	Pair<KeyType, IndexType> l = lft->lastData();
	setFirstIndex(l.index);
	parentNode()->replaceKey(k, l);
	lft->removeLastData();
}

template <typename KeyType, class IndexType> 
BNode<KeyType, IndexType> * BNode<KeyType, IndexType>::parentNode() const
{
	return static_cast<BNode *>(parent());
}
} // end of namespace sdb
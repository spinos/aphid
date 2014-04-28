/*
 *  BNode.cpp
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include "BNode.h"

BNode::BNode(BNode * parent)
{
	for(int i=0;i< MAXPERNODEKEYCOUNT;i++)
        m_data[i].index = NULL;
		
    m_parent = parent;
    m_first = NULL;
	m_numKeys = 0;
	m_isLeaf = false;
}

int BNode::numKeys() const { return m_numKeys; }

bool BNode::isRoot() const { return m_parent == NULL; }

bool BNode::hasChildren() const 
{ 
	if(isLeaf()) return false; 
	return m_first != NULL; 
}

bool BNode::isLeaf() const { return m_isLeaf; }

bool BNode::isFull() const { return m_numKeys == MAXPERNODEKEYCOUNT; }

bool BNode::underflow() const { return m_numKeys < MINPERNODEKEYCOUNT; }

void BNode::setLeaf() { m_isLeaf = true; }

BNode * BNode::nextIndex(int x) const
{
	for(int i= m_numKeys - 1;i >= 0;i--) {
        if(x >= m_data[i].key)
			//std::cout<<"route by key "<<m_data[i].key;
            return (m_data[i].index);
    }
    //return (m_data[m_numKeys-1].index);
	return firstIndex();
}

void BNode::display() const
{
    int i;
    std::cout<<"(";
    for(i=0;i< m_numKeys;i++) {
		std::cout<<m_data[i].key;
		if(i< m_numKeys-1) std::cout<<" ";
	}
    std::cout<<") ";
}

BNode *BNode::firstIndex() const { return m_first; }

int BNode::firstKey() const { return m_data[0].key; }

BNode * BNode::sibling() const
{
	return m_first;
}

BNode * BNode::parent() const
{
	return m_parent;
}

bool BNode::shareSameParent(BNode * another) const
{
	return parent() == another->parent();
}

void BNode::insert(BNode::Pair x)
{
	if(isRoot()) 
		insertRoot(x);
	else if(isLeaf())
		insertLeaf(x);
	else
		insertInterior(x);
}

void BNode::insertRoot(Pair x)
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

void BNode::insertLeaf(Pair x)
{
	if(isFull()) {
		splitLeaf(x);
	}
	else {
		insertData(x);
		//balanceLeaf();
	}
}

void BNode::insertInterior(Pair x)
{
	BNode * n = nextIndex(x.key);
	n->insert(x);
}

void BNode::connectSibling(BNode * another)
{
	//std::cout<<"connect siblings";
	//display();
	//std::cout<<" to ";
	//another->display();
	m_first = another;
}

void BNode::insertData(Pair x)
{	
	int i;
    for(i= m_numKeys - 1;i >= 0 && m_data[i].key > x.key; i--)
        m_data[i+1] = m_data[i];
		
    m_data[i+1]= x;
	//std::cout<<"insert key "<<x.key<<" at "<<i+1<<"\n";
    m_numKeys++;
}

void BNode::splitRoot(Pair x)
{
	//std::cout<<"split root ";
	display();
	BNode * one = new BNode(this); one->setLeaf();
	BNode * two = new BNode(this); two->setLeaf();
	
	partData(x, m_data, one, two, true);
	
	//std::cout<<"into ";
	//one->display();
	//two->display();
	
	m_first = one;
	m_data[0].key = two->firstKey();
	m_data[0].index = two;
	one->connectSibling(two);
	m_numKeys = 1;
	one->balanceLeafLeft();
}

void BNode::splitLeaf(Pair x)
{
	//std::cout<<"split leaf ";
	//display();
	//std::cout<<"into ";
	BNode * oldRgt = sibling();
	BNode * two = new BNode(m_parent); two->setLeaf();

	Pair old[MAXPERNODEKEYCOUNT];
	for(int i=0; i < MAXPERNODEKEYCOUNT; i++)
		old[i] = m_data[i];
		
	m_numKeys = 0;
	partData(x, old, this, two, true);
	//display();
	//two->display();
	
	connectSibling(two);
	if(oldRgt) two->connectSibling(oldRgt);
	
	Pair b;
	b.key = two->firstKey();
	b.index = two;
	m_parent->bounce(b);
	//balanceLeafLeft();
}

void BNode::bounce(Pair b)
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

void BNode::getChildren(BTreeDisplayMap & dst, int level) const
{
	if(isLeaf()) return;
	if(!hasChildren()) return;
	dst[level].push_back(firstIndex());
	for(int i = 0;i < m_numKeys; i++)
		dst[level].push_back(m_data[i].index);
		
	level++;
		
	BNode * n = firstIndex();
	n->getChildren(dst, level);
	for(int i = 0;i < m_numKeys; i++) {
		n = m_data[i].index;
		n->getChildren(dst, level);
	}
}

void BNode::connectChildren()
{
	if(!hasChildren()) return;
	BNode * n = firstIndex();
	n->setParent(this);
	for(int i = 0;i < m_numKeys; i++) {
		n = m_data[i].index;
		n->setParent(this);
	}
}

void BNode::setParent(BNode * parent)
{
	m_parent = parent;
}

void BNode::partRoot(Pair x)
{
	//std::cout<<"part root ";
	//display();
	BNode * one = new BNode(this);
	BNode * two = new BNode(this);
	
	Pair p = partData(x, m_data, one, two);
	
	//std::cout<<"into ";
	//one->display();
	//two->display();
	
	one->setFirstIndex(m_first);
	two->setFirstIndex(p.index);
	
	m_first = one;
	m_data[0].key = p.key;
	m_data[0].index = two;

	m_numKeys = 1;
	
	one->connectChildren();
	two->connectChildren();
}

void BNode::partInterior(Pair x)
{
	std::cout<<"part interior ";
	display();
	BNode * rgt = new BNode(m_parent);
	
	Pair old[MAXPERNODEKEYCOUNT];
	for(int i=0; i < MAXPERNODEKEYCOUNT; i++)
		old[i] = m_data[i];
	
	m_numKeys = 0;
	Pair p = partData(x, old, this, rgt);
	
	std::cout<<"into ";
	display();
	rgt->display();
	
	connectChildren();
	
	rgt->setFirstIndex(p.index);
	rgt->connectChildren();
	
	Pair b;
	b.key = p.key;
	b.index = rgt;
	m_parent->bounce(b);
}

BNode::Pair BNode::partData(Pair x, Pair old[], BNode * lft, BNode * rgt, bool doSplitLeaf)
{
	Pair res, q;
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

void BNode::setFirstIndex(BNode * another)
{
	m_first = another;
}

void BNode::balanceLeaf()
{
	if(!balanceLeafRight()) {
		balanceLeafLeft();
	}
}

bool BNode::balanceLeafRight()
{
	BNode * rgt = sibling();
	if(!rgt) return false;
	
	const Pair s = rgt->firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s, found);
	
	if(!found) return false;
	
	int k = shouldBalance(this, rgt);
	if(k == 0) return false;
	
	Pair old = rgt->firstData();
	if(k < 0) rightData(-k, rgt);
	else rgt->leftData(k, this);
	
	crossed->replaceKey(old, rgt->firstData());
	
	return true;
}

void BNode::balanceLeafLeft()
{
	const Pair s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s, found);
	
	if(!found) return;
	
	BNode * leftSibling = crossed->leafLeftTo(s);
	
	if(leftSibling == this) return;
	
	int k = shouldBalance(leftSibling, this);
	if(k == 0) return;
	
	Pair old = firstData();
	if(k < 0) leftSibling->rightData(-k, this);
	else this->leftData(k, leftSibling);
	
	crossed->replaceKey(old, firstData());
	
	std::cout<<"\nbalanced ";
	leftSibling->display();
	display();
}

BNode * BNode::ancestor(const Pair & x, bool & found) const
{
	if(m_parent->hasKey(x)) {
		found = true;
		return m_parent;
	}
	
	if(m_parent->isRoot()) return NULL;
	return m_parent->ancestor(x, found);
}

bool BNode::hasKey(Pair x) const
{
	for(int i = 0; i < m_numKeys; i++) {
		if(m_data[i].key == x.key)
			return true;
	}
	return false;
}

int BNode::shouldBalance(BNode * lft, BNode * rgt) const
{
	return (lft->numKeys() + rgt->numKeys()) / 2 - lft->numKeys();
}

BNode * BNode::leftTo(const Pair & x) const
{
	if(m_numKeys==1) return firstIndex();
	int i;
	for(i = m_numKeys - 1; i >= 0; i--) {		
        if(m_data[i].key < x.key) {
			return m_data[i].index;
		}
    }
	return firstIndex();
}

BNode * BNode::rightTo(const Pair & x, Pair & k) const
{
	int i;
	for(i = 0; i < m_numKeys; i++) {		
        if(m_data[i].key >= x.key) {
			k = m_data[i];
			return m_data[i].index;
		}
    }
	return NULL;
}

BNode * BNode::leafLeftTo(Pair x)
{
	if(isLeaf()) return this;
	
	BNode * n = leftTo(x);
	return n->leafLeftTo(x);
}

void BNode::rightData(int num, BNode * rgt)
{
	for(int i = 0; i < num; i++) {
		rgt->insertData(lastData());
		removeLastData();
	}
}

void BNode::leftData(int num, BNode * lft)
{
	for(int i = 0; i < num; i++) {
		lft->insertData(firstData());
		removeFirstData();
	}
}

BNode::Pair BNode::lastData() const { return m_data[m_numKeys - 1]; }
BNode::Pair BNode::firstData() const { return m_data[0]; }

void BNode::removeLastData()
{
	m_numKeys--;
}

void BNode::removeFirstData()
{
	for(int i = 0; i < m_numKeys-1; i++) {
		m_data[i] = m_data[i+1];
	}
	m_numKeys--;
}

void BNode::replaceKey(Pair x, Pair y)
{
	for(int i = 0; i < m_numKeys; i++) {
		if(m_data[i].key == x.key) {
			m_data[i].key = y.key;
			return;
		}
	}
}

void BNode::replaceIndex(int n, Pair x)
{
	m_data[n].index = x.index;
}

void BNode::remove(Pair x)
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

void BNode::removeLeaf(const Pair & x)
{
	if(!removeDataLeaf(x)) return;
	if(!underflow()) return;
	
	if(!mergeLeaf())
		balanceLeaf();
}

bool BNode::mergeLeaf()
{
	if(mergeLeafRight())
		return true;
		
	return mergeLeafLeft();
}

bool BNode::mergeLeafRight()
{
	BNode * rgt = sibling();
	if(!rgt) return false;
	
	const Pair s = rgt->firstData();
	
	if(!shouldMerge(this, rgt)) return false;
	
	BNode * up = rgt->parent();

	Pair old = rgt->firstData();
	BNode * oldSibling = rgt->sibling();
	
	rgt->leftData(rgt->numKeys(), this);
	
	delete rgt;
	
	connectSibling(oldSibling);
	
	Pair k = up->dataRightTo(old);
	
	if(m_parent == up) {
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

bool BNode::mergeLeafLeft()
{
	const Pair s = firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s, found);
	
	if(!found) return false;
	
	BNode * leftSibling = crossed->leafLeftTo(s);
	
	if(leftSibling == this) return false;
	
	return leftSibling->mergeLeafRight();
}

bool BNode::removeDataLeaf(const Pair & x)
{
	int i, found = -1;
    for(i= 0;i < m_numKeys; i++) {
        if(m_data[i].key == x.key) {
			found = i;
			break;
		}
	}
	
	if(found < 0) return false;
	
	if(found == m_numKeys - 1) {
		m_numKeys--;
		return true;
	}
	
	for(i= found; i < m_numKeys - 1; i++)
        m_data[i] = m_data[i+1];
		
	if(found == 0) {
		bool c = false;
		BNode * crossed = ancestor(x, c);
		if(c) crossed->replaceKey(x, firstData());
	}
		
    m_numKeys--;
	return true;
}

bool BNode::removeData(const Pair & x)
{
	int i, found = -1;
    for(i= 0;i < m_numKeys; i++) {
        if(m_data[i].key == x.key) {
			found = i;
			break;
		}
	}
	
	if(found < 0) return false;
	
	if(found == 0) 
		m_first = m_data[found].index;
	else
		m_data[found - 1].index = m_data[found].index;

	if(found == m_numKeys - 1) {
		m_numKeys--;
		return true;
	}
	
	for(i= found; i < m_numKeys - 1; i++)
		m_data[i] = m_data[i+1];
		
    m_numKeys--;
	return true;
}

bool BNode::shouldMerge(BNode * lft, BNode * rgt) const
{
	return (lft->numKeys() + rgt->numKeys()) <= MAXPERNODEKEYCOUNT;
}

void BNode::pop(const Pair & x)
{
	if(isRoot()) popRoot(x);
	else popInterior(x);
}

void BNode::popRoot(const Pair & x)
{
	if(numKeys() > 1) removeData(x);
	else {
		BNode * lft = firstIndex();
		m_numKeys = 0;
		mergeData(lft);
		
		if(lft->isLeaf())
			setFirstIndex(NULL);
		else 
			setFirstIndex(lft->firstIndex());
			
		delete lft;
		
		connectChildren();
	}
}

void BNode::popInterior(const Pair & x)
{
	removeData(x);
	if(!underflow()) return;
	std::cout<<"interior underflow! ";display();
	if(!mergeInterior())
		balanceInterior();
}

const BNode::Pair BNode::data(int x) const { return m_data[x]; }

void BNode::mergeData(BNode * another, int start)
{
	const int num = another->numKeys();
	for(int i = start; i < num; i++)
		insertData(another->data(i));
}

const BNode::Pair BNode::dataRightTo(const Pair & x) const
{
	const int num = numKeys();
	for(int i = 0; i < num; i++) {
		if(m_data[i].key >= x.key) return m_data[i];
	}
	return lastData();
}

bool BNode::mergeInterior()
{
	if(mergeInteriorRight()) return true;
	return mergeInteriorLeft();
}

void BNode::balanceInterior()
{
	if(!balanceInteriorRight()) balanceInteriorLeft();
}

bool BNode::shouldInteriorMerge(BNode * lft, BNode * rgt) const
{
	return (lft->numKeys() + rgt->numKeys()) < MAXPERNODEKEYCOUNT;
}

bool BNode::mergeInteriorRight()
{
	Pair k;
	BNode * rgt = m_parent->rightTo(lastData(), k);
	if(!rgt) return false;
	if(!shouldInteriorMerge(this, rgt)) return false;
	
	k.index = rgt->firstIndex();
	
	insertData(k);
	
	rgt->leftData(rgt->numKeys(), this);
	
	delete rgt;
	
	connectChildren();
	
	int ki = m_parent->dataId(k); 
	k.index = this;
	m_parent->setData(ki, k);
	
	m_parent->pop(k);
	return true;
}

bool BNode::mergeInteriorLeft()
{
	BNode * lft =leftInteriorNeighbor();
	if(!lft) return false;
	
	return lft->mergeInteriorRight();
}

BNode * BNode::leftInteriorNeighbor() const
{
	Pair k, j;
	if(!m_parent->dataLeftTo(firstData(), k)) return NULL;
	j = k;
	if(j.index == this)
		m_parent->dataLeftTo(k, j);
	
	if(j.index == this) return NULL;
	return j.index;
}

void BNode::setData(int k, const Pair & x)
{
	m_data[k] = x;
}

int BNode::dataId(const Pair & x) const
{
	for(int i = 0; i < m_numKeys; i++) {		
        if(m_data[i].key >= x.key) {
			return i;
		}
    }
	return -1;
}

bool BNode::dataLeftTo(const Pair & x, Pair & dst) const
{
	for(int i = m_numKeys - 1; i >= 0; i--) {		
        if(m_data[i].key < x.key) {
			dst = m_data[i];
			return true;
		}
    }
	dst.index = firstIndex();
	return false;
}

bool BNode::balanceInteriorRight()
{
	Pair k;
	BNode * rgt = m_parent->rightTo(lastData(), k);
	if(!rgt) return false;

	k.index = rgt->firstIndex();
	
	insertData(k);
	
	m_parent->replaceKey(k, rgt->firstData());
	
	rgt->removeData(rgt->firstData());

	return true;
}

void BNode::balanceInteriorLeft()
{
	BNode * lft = leftInteriorNeighbor();
	if(!lft) return;
	Pair k;
	m_parent->rightTo(lft->lastData(), k);
	k.index = firstIndex();
	insertData(k);
	Pair l = lft->lastData();
	setFirstIndex(l.index);
	m_parent->replaceKey(k, l);
	lft->removeLastData();
}
//~:
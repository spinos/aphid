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
	for(int i=0;i< PERNODEKEYCOUNT;i++)
        m_data[i].index = NULL;
		
    m_parent = parent;
    m_first = NULL;
	m_numKeys = 0;
}

bool BNode::isRoot() const { return m_parent == NULL; }

bool BNode::hasChildren() const { return m_first != NULL; }

bool BNode::isLeaf() const { return m_first == NULL; }

bool BNode::isFull() const { return m_numKeys >= PERNODEKEYCOUNT; }

BNode * BNode::nextIndex(int x) const
{
	//std::cout<<" "<<x<<" ";
	//if(x < m_data[0].key)
    //  return firstIndex();
	
	for(int i= m_numKeys - 1;i >= 0;i--) {
        if(x >= m_data[i].key) {
			//std::cout<<"route by key "<<m_data[i].key;
            return (m_data[i].index);
		}
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
	else 
		insertData(x);
}

void BNode::insertInterior(Pair x)
{
	BNode * n = nextIndex(x.key);
	n->insert(x);
}

void BNode::connect(BNode * another)
{
	m_data[PERNODEKEYCOUNT - 1].index = another;
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
	std::cout<<"split root ";
	display();
	BNode * one = new BNode(this);
	BNode * two = new BNode(this);
	
	splitData(x, m_data, one, two);
	
	std::cout<<"into ";
	one->display();
	two->display();
	
	m_first = one;
	m_data[0].key = two->firstKey();
	m_data[0].index = two;
	one->connect(two);
	m_numKeys = 1;
}

void BNode::splitLeaf(Pair x)
{
	std::cout<<"split leaf ";
	display();
	std::cout<<"into ";
	BNode * two = new BNode(m_parent);
	connect(two);
	Pair old[PERNODEKEYCOUNT];
	for(int i=0; i < PERNODEKEYCOUNT; i++)
		old[i] = m_data[i];
		
	m_numKeys = 0;
	splitData(x, old, this, two);
	display();
	two->display();
	
	Pair b;
	b.key = two->firstKey();
	b.index = two;
	m_parent->bounce(b);
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

void BNode::splitData(Pair x, Pair old[], BNode * lft, BNode * rgt)
{
	BNode * dst = rgt;
	
	int numKeysRight = 0;
	bool inserted = false;
	for(int i = PERNODEKEYCOUNT - 1;i >= 0; i--) {
		if(x.key > old[i].key && !inserted) {
			dst->insertData(x);
			i++;
			inserted = true;
		}
		else
			dst->insertData(old[i]);
			
		numKeysRight++;
		if(numKeysRight >= PERNODEKEYCOUNT / 2 + 1)
			dst = lft;
	}
	
	if(!inserted)
		dst->insertData(x);
}

void BNode::getChildren(BTreeDisplayMap & dst, int level) const
{
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
	std::cout<<"part root ";
	display();
	BNode * one = new BNode(this);
	BNode * two = new BNode(this);
	
	Pair p = partData(x, m_data, one, two);
	
	std::cout<<"into ";
	one->display();
	two->display();
	
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
	
	Pair old[PERNODEKEYCOUNT];
	for(int i=0; i < PERNODEKEYCOUNT; i++)
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

BNode::Pair BNode::partData(Pair x, Pair old[], BNode * lft, BNode * rgt)
{
	Pair res, q;
	BNode * dst = rgt;
	
	int numKeysRight = 0;
	bool inserted = false;
	for(int i = PERNODEKEYCOUNT - 1;i >= 0; i--) {
		if(x.key > old[i].key && !inserted) {
			q = x;
			i++;
			inserted = true;
		}
		else
			q = old[i];
			
		numKeysRight++;
		
		if(numKeysRight == PERNODEKEYCOUNT / 2 + 1) {
			dst = lft;
			res = q;
		}
		else
			dst->insertData(q);
	}
			
	return res;
}

void BNode::setFirstIndex(BNode * another)
{
	m_first = another;
}

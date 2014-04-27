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

#define MAXPERNODEKEYCOUNT 4
#define MINPERNODEKEYCOUNT 2

class BNode;

typedef std::map<int, std::vector<BNode *> > BTreeDisplayMap;

class BNode
{
public:
	struct Pair
	{
		int key;
		BNode * index;
	};

	BNode(BNode * parent = NULL);
	
	bool isRoot() const;
	bool hasChildren() const;
	bool isLeaf() const;
	bool isFull() const;
	
	void insert(Pair x);
    void remove(Pair x);
	
	BNode *firstIndex() const;
	int firstKey() const;
    BNode *nextIndex(int x) const;
	void connectSibling(BNode * another);
    void display() const;
	void getChildren(BTreeDisplayMap & dst, int level) const;
	
	void setFirstIndex(BNode * another);
	
private:
	void insertRoot(Pair x);
	void splitRoot(Pair x);
	
	void insertLeaf(Pair x);
	void splitLeaf(Pair x);
	
	void insertData(Pair x);
	
	void partRoot(Pair x);
	Pair partData(Pair x, Pair old[], BNode * lft, BNode * rgt, bool doSplitLeaf = false);
	
	void partInterior(Pair x);
	
	void insertInterior(Pair x);
	void connectChildren();
	void setParent(BNode * parent);
	
	void bounce(Pair b);
	BNode * sibling() const;
	BNode * parent() const;
	bool shareSameParent(BNode * another) const;
	
	void balanceLeaf();
	bool balanceLeafRight();
	void balanceLeafLeft();
	void rightData(int num, BNode * rgt);
	void leftData(int num, BNode * lft);
	
	Pair lastData() const;
	Pair firstData() const;
	
	void removeLastData();
	void removeFirstData();
	
	int numKeys() const;
	
	void setLeaf();
	void replaceKey(Pair x, Pair y);
	
	BNode * ancestor(const Pair & x, bool & found) const;
	BNode * leftTo(const Pair & x) const;
	BNode * leafLeftTo(Pair x);
	
	bool hasKey(Pair x) const;
	int shouldBalance(BNode * lft, BNode * rgt) const;
	
	void removeLeaf(const Pair & x);
	bool removeData(const Pair & x);
	
private:
    Pair m_data[MAXPERNODEKEYCOUNT];
    BNode *m_first;
	BNode *m_parent;
    int m_numKeys;
	bool m_isLeaf;
};
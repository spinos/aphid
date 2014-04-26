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
#define PERNODEINDEXCOUNT 5
#define PERNODEKEYCOUNT 4

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
    void bounce(Pair b);
	
	BNode *firstIndex() const;
	int firstKey() const;
    BNode *nextIndex(int x) const;
	void connect(BNode * another);
    void display() const;
	void getChildren(BTreeDisplayMap & dst, int level) const;
	
	void setFirstIndex(BNode * another);
	
private:
	void insertRoot(Pair x);
	void splitRoot(Pair x);
	
	void insertLeaf(Pair x);
	void splitLeaf(Pair x);
	
	void insertData(Pair x);
	void splitData(Pair x, Pair old[], BNode * lft, BNode * rgt);
	
	void partRoot(Pair x);
	Pair partData(Pair x, Pair old[], BNode * lft, BNode * rgt);
	
	void partInterior(Pair x);
	
	void insertInterior(Pair x);
	void connectChildren();
	void setParent(BNode * parent);
	
    BNode *m_parent;
    int m_numKeys;
    Pair m_data[PERNODEKEYCOUNT];
    BNode *m_first;
};
/*
 *  BTree.h
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <BNode.h>

namespace sdb {

class BTree
{
public:
    BTree();
    void insert(int x);
	void remove(int x);
    void display();
	
private:
	void displayLevel(const int & level, const std::vector<Entity *> & nodes);
	
	BNode<int> * m_root;
};
} // end of namespace sdb

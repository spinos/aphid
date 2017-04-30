/*
 *  BTree.h
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <sdb/BNode.h>
#include <sdb/List.h>
namespace aphid {
namespace sdb {

class BTree
{
public:
    BTree();
    void insert(int x);
	void remove(int x);
	bool find(int x);
    void display();
	void displayLeaves();
	
private:
	void displayLevel(const int & level, const std::vector<Entity *> & nodes);
	
	BNode<int> * m_root;
};
} // end namespace sdb
}

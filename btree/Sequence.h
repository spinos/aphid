/*
 *  Sequence.h
 *  btree
 *
 *  Created by jian zhang on 5/9/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BNode.h>
namespace sdb {
template<typename T>
class Sequence : public Entity {
public:
	Sequence(Entity * parent = NULL) : Entity(parent) {
		m_root = new BNode<T>;
	}
	
	virtual ~Sequence() {
		delete m_root;
	}
	
	void insert(const T & x) { 
		m_root->insert(x);
	}
	
	void remove(const T & x) {
		m_root->remove(x);
	}
	
	bool find(const T & x) {
		Pair<Entity *, Entity> g = m_root->find(x);
		if(!g.index) return false;
		return true;
	}
	
	int size() {
		int c = 0;
		BNode<T> * current = m_root->firstLeaf();
		while(current) {
			c += current->numKeys();
			current = current->nextLeaf();
		}
		return c;
	}

private:
	
private:
	BNode<T> * m_root;
};
} //end namespace sdb

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
#include <deque>
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
	
	Pair<T, Entity> * insert(const T & x) { 
		return m_root->insert(x);
	}
	
	void remove(const T & x) {
		m_root->remove(x);
		m_current = NULL;
	}
	
	bool find(const T & x) {
		Pair<Entity *, Entity> g = m_root->find(x);
		if(!g.index) return false;
		return true;
	}
	
	Pair<Entity *, Entity> findEntity(const T & x) {
		return m_root->find(x);
	}
	
	void begin() {
		beginLeaf();
		if(leafEnd()) return;
		m_currentData = 0;
	}
	
	void next() {
		m_currentData++;
		if(m_currentData == leafSize()) {
			nextLeaf();
			if(leafEnd()) return;
			m_currentData = 0;
		} 
	}
	
	const bool end() const {
		return leafEnd();
	}
	
	int size() {
		int c = 0;
		beginLeaf();
		while(!leafEnd()) {
			c += leafSize();
			nextLeaf();
		}
		return c;
	}
	
	void clear() {
		delete m_root;
		m_root = new BNode<T>;
	}

protected:	
	void beginLeaf() {
		m_current = m_root->firstLeaf();
	}
	
	void nextLeaf() {
		m_current = static_cast<BNode<T> *>(m_current->sibling()); 
	}
	
	const bool leafEnd() const {
		return (m_current == NULL || m_root->numKeys() < 1);
	}
	
	const int leafSize() const {
		if(!m_current) return 0;
		return m_current->numKeys();
	}
	
	Entity * currentIndex() const {
		if(!m_current) return NULL;
		return m_current->index(m_currentData);
	}
	
	const T currentKey() const {
		return m_current->key(m_currentData);
	}
	
	std::deque<T> allKeys() {
		typename std::deque<T> r;
		begin();
		while(!end()) {
			r.push_back(currentKey());
			next();
		}
		return r;
	}

private:
	
private:
	BNode<T> * m_root;
	BNode<T> * m_current;
	int m_currentData;
};
} //end namespace sdb

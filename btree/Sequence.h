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
	
	Pair<T, Entity> * insert(const T & x) { 
		return m_root->insert(x);
	}
	
	void remove(const T & x) {
		m_root->remove(x);
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
		m_dataEnd = leafSize();
	}
	
	void next() {
		m_currentData++;
		if(m_currentData == m_dataEnd) {
			nextLeaf();
			if(leafEnd()) return;
			m_currentData = 0;
			m_dataEnd = leafSize();
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
		return m_current->numKeys();
	}
	
	Entity * currentIndex() const {
		return m_current->index(m_currentData);
	}
	
	const T currentKey() const {
		return m_current->key(m_currentData);
	}

private:
	
private:
	BNode<T> * m_root;
	BNode<T> * m_current;
	int m_currentData, m_dataEnd;
};
} //end namespace sdb

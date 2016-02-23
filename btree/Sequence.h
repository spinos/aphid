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

namespace aphid {

namespace sdb {

class MatchFunction {
public:
	enum Condition {
		mExact = 0,
		mLequal = 1
	};
};
	
template<typename T>
class Sequence : public Entity {

	bool m_isDataExternal;
	
public:
	Sequence(Entity * parent = NULL) : Entity(parent),
	m_isDataExternal(false)
	{
		m_root = new BNode<T>();
	}
	
	virtual ~Sequence() {
		if(!m_isDataExternal) delete m_root;
	}
	
	Pair<T, Entity> * insert(const T & x) { 
#if 0
		std::cout<<"\n seq insert"<<x;
		Pair<T, Entity> * e = m_root->insert(x);
		std::cout<<"\n res"<<e->index;
		return e;
#else
		return m_root->insert(x);
#endif
	}
	
	void remove(const T & x) {
		m_root->remove(x);
		m_current = NULL;
	}
	
	bool find(const T & x) {
		if(m_root->numKeys() < 1) return false;
		Pair<Entity *, Entity> g = m_root->find(x);
		if(!g.index) return false;

		return true;
	}
	
	Pair<Entity *, Entity> findEntity(const T & x, MatchFunction::Condition mf = MatchFunction::mExact, T * extraKey = NULL) const
	{
		SearchResult sr;
		Pair<Entity *, Entity> g = m_root->find(x);
/// exact
		if(g.index) {
			if(extraKey) *extraKey = x;
			return g;
		}
		
        if(mf == MatchFunction::mExact) return g;
		
		BNode<T> * lastSearchNode = static_cast<BNode<T> *>(g.key);
		
		g = lastSearchNode->findInNode(x, &sr);
			
			//std::cout<<"\n last search "<<*m_lastSearchNode
			//<<" sr.low "<<sr.low<<" sr.high "<<sr.high<<" sr.found "<<sr.found;

		if(mf == MatchFunction::mLequal) {
			if(lastSearchNode->key(sr.high) < x) {
				g.key = lastSearchNode;
				g.index = lastSearchNode->index(sr.high);
				if(extraKey) *extraKey = lastSearchNode->key(sr.high);
			}
			else {
				g.key = lastSearchNode;
				g.index = lastSearchNode->index(sr.low);
				if(extraKey) *extraKey = lastSearchNode->key(sr.low);
			}
		}
		
		return g;
	}
    
	void begin() {
		beginLeaf();
		if(leafEnd()) return;
		m_currentData = 0;
	}
	
	bool beginAt(const T & x) 
	{
        Pair<Entity *, Entity> g = findEntity(x);
		m_current = static_cast<BNode<T> *>(g.key);
		if(!m_current) return false;
        SearchResult sr;
		g = m_current->findInNode(x, &sr);
		m_currentData = sr.found;
		return true;
	}
	
	void next() {
		m_currentData++;
		if(m_currentData >= leafSize()) {
			nextLeaf();
			if(leafEnd()) return;
			m_currentData = 0;
		} 
	}
	
	const bool end() const {
		return leafEnd();
	}
	
	const T key() const {
		return m_current->key(m_currentData);
	}
	
	void setToEnd() {
		m_current = NULL;
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
	
	virtual void clear() {
		delete m_root;
		m_root = new BNode<T>();
	}
	
	bool intersect(Sequence * another)
	{
		begin();
		while(!end()) {
			if(another->find(currentKey())) return true;
			next();
		}
		return false;
	}
	
	bool isEmpty() const
	{ return m_root->numKeys() < 1; }
	
	void verbose() {
		std::cout<<"\n sequence root node "<<*m_root;
	}

    virtual void display();
	
	void setDataExternal()
	{ m_isDataExternal = true; }

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
	void displayLevel(const int & level, const std::vector<Entity *> & nodes);

private:
	BNode<T> * m_root;
	BNode<T> * m_current;
	int m_currentData;
};

template<typename T>
void Sequence<T>::display()
{
    std::cout<<"\n sequence display "
    <<"\n root "<<*m_root;
	
	std::map<int, std::vector<Entity *> > nodes;
	m_root->getChildren(nodes, 1);
	
	std::map<int, std::vector<Entity *> >::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it)
		displayLevel((*it).first, (*it).second);
}

template<typename T>
void Sequence<T>::displayLevel(const int & level, const std::vector<Entity *> & nodes)
{
	std::cout<<"\n level: "<<level<<" ";
	std::vector<Entity *>::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it)
		std::cout<<*(static_cast<BNode<T> *>(*it));
}

} //end namespace sdb

}
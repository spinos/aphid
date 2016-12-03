/*
 *  Sequence.h
 *  btree
 *
 *  Created by jian zhang on 5/9/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <sdb/BNode.h>
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
		return m_root->insert(x);
	}
	
	void remove(const T & x) {
		m_root->remove(x);
		m_current = NULL;
	}
	
	bool findKey(const T & x) {
		if(m_root->numKeys() < 1) return false;
		Pair<Entity *, Entity> g = m_root->find(x);
		BNode<T> * lastSearchNode = static_cast<BNode<T> *>(g.key);
		
		SearchResult sr;
		lastSearchNode->findInNode(x, &sr);

		return sr.found > -1;
	}
	
	bool find(const T & x) {
		if(m_root->numKeys() < 1) return false;
		Pair<Entity *, Entity> g = m_root->find(x);
		if(!g.index) return false;

		return true;
	}
	
	Pair<Entity *, Entity> findEntity(const T & x, MatchFunction::Condition mf = MatchFunction::mExact, T * extraKey = NULL) const
	{
		Pair<Entity *, Entity> g = m_root->find(x);
/// exact
		if(g.index) {
			if(extraKey) *extraKey = x;
			return g;
		}
		
        if(mf == MatchFunction::mExact) return g;
		
		BNode<T> * lastSearchNode = static_cast<BNode<T> *>(g.key);
		
		SearchResult sr;
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
		if(m_currentData == leafSize()) {
			nextLeaf();
			if(leafEnd()) return;
			m_currentData = 0;
		} 
	}
	
	const bool end() const {
		return leafEnd();
	}
	
	const T & key() const {
		return m_current->key(m_currentData);
	}
	
	void setToEnd() {
		m_current = NULL;
	}
	
	int size() {
		int c = 0;
		try {
		beginLeaf();
		while(!leafEnd()) {
			c += leafSize();
			nextLeaf();
		}
		} catch (const char * ex) {
			std::cerr<<"Sequence size caught "<<ex;
		} catch (...) {
			std::cerr<<"Sequence size caught something";
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
	
	int numLeaf() {
		int nn = 0;
		beginLeaf();
		while(!leafEnd()) {
			nn++;
			nextLeaf();
		}
		return nn;
	}

	void dbgFind(const T & x);
	
	std::deque<T> allKeys() {
		typename std::deque<T> r;
		begin();
		while(!end()) {
			r.push_back(currentKey());
			next();
		}
		return r;
	}
	
	bool dbgCheck();
	
	void printLeafSize();
	
protected:	
	void beginLeaf() {
		m_current = m_root->firstLeaf();
	}
	
	void nextLeaf() {
		m_current = m_current->nextLeaf();
	}
	
	const bool leafEnd() const {
		if(m_root->numKeys() < 1) return true;
		return (m_current == NULL);
	}
	
	const int leafSize() const {
		if(!m_current) return 0;
		return m_current->numKeys();
	}
	
	Entity * currentIndex() const {
		if(m_current == NULL) return NULL;
		return m_current->index(m_currentData);
	}
	
	const T currentKey() const {
		return m_current->key(m_currentData);
	}
	
private:
	void displayLevel(const int & level, const std::vector<Entity *> & nodes);
	bool dbgCheckLayer(const int & level, const std::vector<Entity *> & nodes);

private:
	BNode<T> * m_root;
	BNode<T> * m_current;
	int m_currentData;
};

template<typename T>
void Sequence<T>::display()
{
    std::cout<<"\n sequence display "
    <<"\n  "<<*m_root;
	
    if(!m_root->hasChildren() ) return;
    m_root->dbgDown();
    
	std::map<int, std::vector<Entity *> > nodes;
	m_root->getChildren(nodes, 1);
	
	std::map<int, std::vector<Entity *> >::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it)
		displayLevel((*it).first, (*it).second);
}

template<typename T>
void Sequence<T>::displayLevel(const int & level, const std::vector<Entity *> & nodes)
{
	std::cout<<"\n level: "<<level<<" n "<<nodes.size()<<" ";
	std::vector<Entity *>::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it) {
	    BNode<T> * n = static_cast<BNode<T> *>(*it);
		std::cout<<"\n"<<*n;
		n->dbgDown();
	}
}

template<typename T>
void Sequence<T>::dbgFind(const T & x)
{
	m_root->dbgFind(x);
}

template<typename T>
bool Sequence<T>::dbgCheck()
{ 
	if( !m_root->dbgLinks() ) return false;
	
	std::map<int, std::vector<Entity *> > nodes;
	m_root->getChildren(nodes, 1);
	
	std::map<int, std::vector<Entity *> >::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it) {
		if(!dbgCheckLayer((*it).first, (*it).second)) return false;
	}
		
	return true;
}

template<typename T>
bool Sequence<T>::dbgCheckLayer(const int & level, const std::vector<Entity *> & nodes)
{
    int nl = 0;
    std::vector<Entity *>::const_iterator it = nodes.begin();
	for(; it != nodes.end(); ++it) {
		BNode<T> * n = static_cast<BNode<T> *>(*it);
		if(n->isLeaf() ) {
		    nl += n->numKeys();
		    continue;
		}
		if(!n->dbgLinks(true) ) return false;
	}
	
	if(nl > 0) {
	    if(nl != size()) {
	        std::cout<<"\n wrong size "<<nl
	            <<" != "<<size();
	            return false;
	    }
	}
	return true;
}

template<typename T>
void Sequence<T>::printLeafSize()
{
	std::cout<<"\n";
	int c = 0, i=0;
	beginLeaf();
	while(!leafEnd()) {
		std::cout<<" leaf["<<i++<<"]"<< leafSize();
		c+= leafSize();
		nextLeaf();
	}
	std::cout<<" total "<< c;
}

} //end namespace sdb

}
/*
 *  LimitedSequence.h
 *  sdb
 *
 *  in one limited trunk, no hierarchy
 *
 *  Created by jian zhang on 3/7/17.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_SDB_LIMITED_SEQUENCE_H
#define APH_SDB_LIMITED_SEQUENCE_H

#include <sdb/BNode.h>

namespace aphid {

namespace sdb {
	
template<typename T, int MaxNKey>
class LimitedSequence : public Entity {

typedef BNode<T, MaxNKey> RootTyp;

	RootTyp * m_root;
	int m_currentData;
	bool m_isDataExternal;
	
public:
	LimitedSequence(Entity * parent = NULL);
	virtual ~LimitedSequence();
	
	const int & size() const;
	bool isEmpty() const;
	bool isFull() const;
	
	void setDataExternal();
	
	const T & key() const;
	
	void begin();
	void next();
	bool end() const;
	
	Pair<T, Entity> * insert(const T & x);
	
	bool findKey(const T & x);
	
	bool find(const T & x) const;
	
	Pair<Entity *, Entity> findEntity(const T & x) const;
	
	bool intersect(LimitedSequence * another);
	void verbose();
	
	virtual void remove(const T & x);
	virtual void clear();
	virtual void display();
	
protected:
	
	Entity * currentIndex() const;
	const T & currentKey() const;
	
private:
	
};

template<typename T, int MaxNKey>
LimitedSequence<T, MaxNKey>::LimitedSequence(Entity * parent) : Entity(parent),
m_isDataExternal(false)
{
	m_root = new RootTyp();
}
	
template<typename T, int MaxNKey>
LimitedSequence<T, MaxNKey>::~LimitedSequence() 
{
	delete m_root;
}

template<typename T, int MaxNKey>
const int & LimitedSequence<T, MaxNKey>::size() const 
{
    return m_root->numKeys();
}

template<typename T, int MaxNKey>
Pair<T, Entity> * LimitedSequence<T, MaxNKey>::insert(const T & x) 
{ 
    if(isFull() ) {
        return NULL;   
    }
    return m_root->insert(x);
}
	
template<typename T, int MaxNKey>
void LimitedSequence<T, MaxNKey>::remove(const T & x) 
{
    m_root->remove(x);
}

template<typename T, int MaxNKey>
void LimitedSequence<T, MaxNKey>::setDataExternal()
{ m_isDataExternal = true; }	

template<typename T, int MaxNKey>
bool LimitedSequence<T, MaxNKey>::isEmpty() const
{ return m_root->numKeys() < 1; }

template<typename T, int MaxNKey>
bool LimitedSequence<T, MaxNKey>::isFull() const
{ return m_root->numKeys() >= MaxNKey; }

template<typename T, int MaxNKey>
const T & LimitedSequence<T, MaxNKey>::key() const 
{
	return m_root->key(m_currentData);
}

template<typename T, int MaxNKey>
void LimitedSequence<T, MaxNKey>::begin() 
{
    m_currentData = 0;
}
	
template<typename T, int MaxNKey>
void LimitedSequence<T, MaxNKey>::next() 
{
    m_currentData++; 
}
	
template<typename T, int MaxNKey>
bool LimitedSequence<T, MaxNKey>::end() const 
{
    return m_currentData >= size();
}

template<typename T, int MaxNKey>
void LimitedSequence<T, MaxNKey>::verbose() 
{
	std::cout<<"\n LimitedSequence root node "<<*m_root;
}

template<typename T, int MaxNKey>
bool LimitedSequence<T, MaxNKey>::intersect(LimitedSequence * another)
{
    begin();
    while(!end()) {
        if(another->find(currentKey() ) ) {
            return true;
        }
        
        next();
    }
    return false;
}

template<typename T, int MaxNKey>
void LimitedSequence<T, MaxNKey>::clear() 
{
    delete m_root;
    m_root = new RootTyp();
}
	
template<typename T, int MaxNKey>
void LimitedSequence <T, MaxNKey>::display()
{
    std::cout<<"\n LimitedSequence display "
    <<"\n  "<<*m_root;
}

template<typename T, int MaxNKey>
Entity * LimitedSequence <T, MaxNKey>::currentIndex() const 
{
    return m_root->index(m_currentData);
}

template<typename T, int MaxNKey>
const T & LimitedSequence <T, MaxNKey>::currentKey() const 
{
    return m_root->key(m_currentData);
}

template<typename T, int MaxNKey>
bool LimitedSequence <T, MaxNKey>::findKey(const T & x) {
    if(isEmpty()) {
        return false;
    }
    
    Pair<Entity *, Entity> g = m_root->find(x);
    if(!g.index) return false;

    return true;
}
	
template<typename T, int MaxNKey>
bool LimitedSequence <T, MaxNKey>::find(const T & x) const
{
    if(isEmpty()) {
        return false;
    }

	SearchResult sr;
	m_root->findInNode(x, &sr);

	return sr.found > -1;

}
	
template<typename T, int MaxNKey>
Pair<Entity *, Entity> LimitedSequence <T, MaxNKey>::findEntity(const T & x) const
{    
    Pair<Entity *, Entity> g = m_root->find(x);
    
    return g;
}

} //end namespace sdb

}
#endif
/*
 *  BaseArray.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <vector>
#include <iostream>
#define BASEARRNUMELEMPERBLK 32768
#define BASEARRNUMELEMPERBLKM1 32767
#define BASEARRNUMELEMPERBLKL2 15

template<typename T>
class BaseArrayBlock {
public:
	BaseArrayBlock(BaseArrayBlock * parent = NULL);
	~BaseArrayBlock();
	
	unsigned globalIndex() const;
	void setGlobalIndex(unsigned x);
	
	void connectToRight(BaseArrayBlock * another);
	void disconnect();
	
	bool hasParent() const;
	BaseArrayBlock * parent();
	bool hasChild() const;
	BaseArrayBlock * child();
	
	T * data();
	T & cdata() const;
	void setData(const T & x);
	
	void begin();
	void next();
	bool end();
	
	unsigned index() const;
	void setIndex(unsigned x);
	
	void chainBegin();
	
private:
	BaseArrayBlock * m_parent;
	BaseArrayBlock * m_child;
	T * m_data;
	unsigned m_loc;
	unsigned m_globalIndex;
};

template<typename T>
BaseArrayBlock<T>::BaseArrayBlock(BaseArrayBlock * parent)
{
	m_data = new T[BASEARRNUMELEMPERBLK];
	m_parent = parent;
	m_child = NULL;
	m_loc = 0;
	m_globalIndex = 0;
}

template<typename T>
BaseArrayBlock<T>::~BaseArrayBlock()
{
	if(m_child) delete m_child;
	delete m_data;
}

template<typename T>
unsigned BaseArrayBlock<T>::globalIndex() const
{ return m_globalIndex; }

template<typename T>
void BaseArrayBlock<T>::setGlobalIndex(unsigned x)
{ m_globalIndex = x; }

template<typename T>
void BaseArrayBlock<T>::connectToRight(BaseArrayBlock * another)
{ 
	m_child = another; 
	another->setGlobalIndex(m_globalIndex + BASEARRNUMELEMPERBLK);
}

template<typename T>
void BaseArrayBlock<T>::disconnect()
{
	if(m_child) delete m_child;
	m_child = NULL;
}

template<typename T>
T * BaseArrayBlock<T>::data()
{ return &m_data[m_loc]; }

template<typename T>
T & BaseArrayBlock<T>::cdata() const
{ return m_data[m_loc]; }

template<typename T>
void BaseArrayBlock<T>::setData(const T & x)
{ m_data[m_loc] = x; }

template<typename T>
bool BaseArrayBlock<T>::hasParent() const
{ return (m_parent != NULL); }

template<typename T>
BaseArrayBlock<T> * BaseArrayBlock<T>::parent()
{ return m_parent; }

template<typename T>
bool BaseArrayBlock<T>::hasChild() const
{ return m_child != NULL; }

template<typename T>
BaseArrayBlock<T> * BaseArrayBlock<T>::child()
{ return m_child; }

template<typename T>
void BaseArrayBlock<T>::begin()
{ m_loc = 0; }

template<typename T>
void BaseArrayBlock<T>::next()
{ m_loc++; }

template<typename T>
bool BaseArrayBlock<T>::end()
{ return m_loc == BASEARRNUMELEMPERBLKM1; }

template<typename T>
unsigned BaseArrayBlock<T>::index() const
{ return m_loc; }

template<typename T>
void BaseArrayBlock<T>::setIndex(unsigned x)
{ m_loc = x; }

template<typename T>
void BaseArrayBlock<T>::chainBegin()
{ 
	begin();
	if(m_child) m_child->chainBegin();
}


template<typename T>
class BaseArray {
public:
	BaseArray();
	virtual ~BaseArray();

	void clear();
	void initialize();
	
	void expandBy(unsigned size);
	
	void begin();
	void next();
	bool end() const;
	
	unsigned index() const;
	void setIndex(unsigned index);
	
	void setValue(const T & x);
	T & value() const;
	T * current();
	T * at(unsigned index);
	
	unsigned capacity() const;
	unsigned numBlocks() const;
	
	void verbose() const;
	
protected:
	void nextBlock();
	
private:
	BaseArrayBlock<T> * m_root;
	BaseArrayBlock<T> * m_lastBlock;
	unsigned m_capacity;
};


template<typename T>
BaseArray<T>::BaseArray() 
{
	m_root = new BaseArrayBlock<T>;
	initialize();
}

template<typename T>
BaseArray<T>::~BaseArray() 
{ delete m_root; }

template<typename T>
void BaseArray<T>::clear() 
{
	m_root->disconnect();
	initialize();
}

template<typename T>
void BaseArray<T>::initialize()
{
	m_capacity = BASEARRNUMELEMPERBLK;
	begin();
}

template<typename T>
void BaseArray<T>::begin()
{
	m_root->chainBegin();
	m_lastBlock = m_root;
}

template<typename T>
void BaseArray<T>::next()
{
	if(m_lastBlock->end())
		nextBlock();
	else
		m_lastBlock->next();
}

template<typename T>
void BaseArray<T>::nextBlock()	
{
	m_lastBlock = m_lastBlock->child();
	// m_lastBlock->begin();
}

template<typename T>
bool BaseArray<T>::end() const
{
	return (m_lastBlock->end() 
				&& (!m_lastBlock->hasChild()));
}

template<typename T>
void BaseArray<T>::expandBy(unsigned size)
{
	int overflown = index() + 1 + size - capacity();
	if(overflown > 0) {
		unsigned blockToCreate = (overflown >> BASEARRNUMELEMPERBLKL2) + 1;
		BaseArrayBlock<T> * head = m_lastBlock;
		for(unsigned i = 0; i < blockToCreate; i++) {
			BaseArrayBlock<T> * tail = new BaseArrayBlock<T>(head);
			head->connectToRight(tail);
			head = tail;
			
			m_capacity += BASEARRNUMELEMPERBLK;
		}
	}
}

template<typename T>
void BaseArray<T>::setValue(const T & x)
{ m_lastBlock->setData(x); }

template<typename T>
T * BaseArray<T>::current()
{ return m_lastBlock->data(); }

template<typename T>
T & BaseArray<T>::value() const
{ return m_lastBlock->cdata(); }

template<typename T>
unsigned BaseArray<T>::index() const
{
	return m_lastBlock->globalIndex() 
			+ m_lastBlock->index();
}

template<typename T>
void BaseArray<T>::setIndex(unsigned index)
{
	unsigned y = index >> BASEARRNUMELEMPERBLKL2;
	unsigned b = y * BASEARRNUMELEMPERBLK;

	if(m_lastBlock->globalIndex() > b ) {
		while(m_lastBlock->hasParent() && m_lastBlock->globalIndex() > b) {
			m_lastBlock = m_lastBlock->parent();
		}
	}
	else if(m_lastBlock->globalIndex() < b ) {
		while(m_lastBlock->hasChild() && m_lastBlock->globalIndex() < b) {
			m_lastBlock = m_lastBlock->child();
		}
	}
	
	m_lastBlock->setIndex(index - b);
}

template<typename T>
T * BaseArray<T>::at(unsigned index)
{
	setIndex(index);
	return current();
}

template<typename T>
unsigned BaseArray<T>::capacity() const 
{ return m_capacity; }

template<typename T>
unsigned BaseArray<T>::numBlocks() const
{ return m_capacity / BASEARRNUMELEMPERBLK; }

template<typename T>
void BaseArray<T>::verbose() const
{
	std::cout<<"base array:\n";
    std::cout<<"elem size "<<sizeof(T)<<"\n";
    std::cout<<"elem per blk "<<BASEARRNUMELEMPERBLK<<"\n";
	std::cout<<"num blk "<<numBlocks()<<"\n";
    std::cout<<"capacity "<<capacity()<<"\n";
    std::cout<<"current index "<<index()<<"\n";
}
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
	BaseArrayBlock();
	~BaseArrayBlock();
	
	void connectTo(BaseArrayBlock * another);
	void disconnect();
	BaseArrayBlock * sibling();
	T * data();
	T cdata() const;
	void setData(const T & x);
	
	void begin();
	void next();
	bool end();
	
	unsigned index() const;
	void setIndex(unsigned x);
	
private:
	BaseArrayBlock * m_sibling;
	T * m_data;
	int m_loc;
};

template<typename T>
BaseArrayBlock<T>::BaseArrayBlock()
{
	m_data = new T[BASEARRNUMELEMPERBLK];
	m_sibling = NULL;
	m_loc = 0;
}

template<typename T>
BaseArrayBlock<T>::~BaseArrayBlock()
{
	if(m_sibling) delete m_sibling;
	delete m_data;
}

template<typename T>
void BaseArrayBlock<T>::connectTo(BaseArrayBlock * another)
{ m_sibling = another; }

template<typename T>
void BaseArrayBlock<T>::disconnect()
{
	if(m_sibling) delete m_sibling;
	m_sibling = NULL;
}

template<typename T>
T * BaseArrayBlock<T>::data()
{ return &m_data[m_loc]; }

template<typename T>
T BaseArrayBlock<T>::cdata() const
{ return m_data[m_loc]; }

template<typename T>
void BaseArrayBlock<T>::setData(const T & x)
{ m_data[m_loc] = x; }

template<typename T>
BaseArrayBlock<T> * BaseArrayBlock<T>::sibling()
{ return m_sibling; }

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
	T value() const;
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
	unsigned m_currentBlock;
	unsigned m_numBlocks;
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
	m_numBlocks = 1;
	m_capacity = BASEARRNUMELEMPERBLK;
	begin();
}

template<typename T>
void BaseArray<T>::begin()
{
	m_currentBlock = 0;
	m_lastBlock = m_root;
	m_lastBlock->begin();
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
	m_lastBlock = m_lastBlock->sibling();
	m_lastBlock->begin();
	m_currentBlock++;
}

template<typename T>
bool BaseArray<T>::end() const
{
	return (m_lastBlock->end() 
				&& m_currentBlock == m_numBlocks-1);
}

template<typename T>
void BaseArray<T>::expandBy(unsigned size)
{
	int overflown = index() + 1 + size - capacity();
	if(overflown > 0) {
		unsigned blockToCreate = (overflown >> BASEARRNUMELEMPERBLKL2) + 1;
		BaseArrayBlock<T> * head = m_lastBlock;
		for(unsigned i = 0; i < blockToCreate; i++) {
			BaseArrayBlock<T> * tail = new BaseArrayBlock<T>;
			head->connectTo(tail);
			head = tail;
			
			m_numBlocks++;
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
T BaseArray<T>::value() const
{ return m_lastBlock->cdata(); }

template<typename T>
unsigned BaseArray<T>::index() const
{
	return (m_currentBlock * BASEARRNUMELEMPERBLK) 
			+ m_lastBlock->index();
}

template<typename T>
void BaseArray<T>::setIndex(unsigned index)
{
	unsigned y = index >> BASEARRNUMELEMPERBLKL2;
	unsigned x = index - ( y * BASEARRNUMELEMPERBLK );
	
	if(m_currentBlock != y) {
		m_lastBlock = m_root;
		m_currentBlock = 0;
		unsigned i = 0;
		for(;i<y;i++) {
			m_lastBlock = m_lastBlock->sibling();
			m_currentBlock++;
		}
	}
	
	m_lastBlock->setIndex(x);
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
{ return m_numBlocks; }

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
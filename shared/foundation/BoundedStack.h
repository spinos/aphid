/*
 *  BoundedStack.h
 *  
 *
 *  Created by jian zhang on 8/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_BOUNDED_STACK_H
#define APH_BOUNDED_STACK_H

namespace aphid {

template<typename T, int Nlimit>
class BoundedStack {

	T m_data[Nlimit];
	int m_topLoc;
	
public:
	BoundedStack();
/// reset top to 0	
	void clear();
/// put x on top
	bool push(T x);
/// take item on top
	bool pop(T& y);
/// location of top
	int size() const;
	
protected:

private:
};

template<typename T, int Nlimit>
BoundedStack<T, Nlimit>::BoundedStack() :
m_topLoc(0)
{}

template<typename T, int Nlimit>
void BoundedStack<T, Nlimit>::clear()
{ m_topLoc = 0; }

template<typename T, int Nlimit>
bool BoundedStack<T, Nlimit>::push(T x)
{ 
	if(m_topLoc == Nlimit)
		return false;
		
	m_data[m_topLoc] = x;
	m_topLoc++;
}

template<typename T, int Nlimit>
bool BoundedStack<T, Nlimit>::pop(T& y)
{ 
	if(m_topLoc == 0)
		return false;
		
	m_topLoc--;
	y = m_data[m_topLoc];
	return true;
}

template<typename T, int Nlimit>
int BoundedStack<T, Nlimit>::size() const
{ return m_topLoc; }

}
#endif

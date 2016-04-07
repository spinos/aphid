#pragma once
#include <iostream>
#include <Entity.h>
#include <Types.h>

namespace aphid {

namespace sdb {

struct SearchResult
{
	int found, low, high;
};

template <typename KeyType, int MaxNKey>
class KeyNData {

    Pair<KeyType, Entity> m_data[MaxNKey];
    int m_numKeys;
    
public :
    KeyNData();
    virtual ~KeyNData();
    
	const int & numKeys() const;
	const KeyType & key(const int & i) const;
	Entity * index(const int & i) const;
	
protected:
    bool isFull() const;
	void reduceNumKeys();
    void increaseNumKeys();
    void setNumKeys(int x);
    bool isSingular() const;
    bool isEmpty() const;
    const Pair<KeyType, Entity> & data(int x) const; 
    Pair<KeyType, Entity> * dataR(int x);
	const KeyType & firstKey() const;
	const KeyType & lastKey() const;
	void insertData(Pair<KeyType, Entity> x);
	int insertKey(const KeyType & x);
	void setKey(int k, const KeyType & x);
	void setIndex(int k, Entity * x);
	void setData(int k, const Pair<KeyType, Entity> & x);
	bool hasKey(const KeyType & x) const;
	const Pair<KeyType, Entity> & firstData() const;
	const Pair<KeyType, Entity> & lastData() const;
	void removeFirstData();
	void removeFirstData1();
	void removeLastData();
	void replaceKey(const KeyType & x, const KeyType & y);
	void replaceIndex(int n, Pair<KeyType, Entity> x);
	bool removeKeyAndData(const KeyType & x);
	const Pair<KeyType, Entity> dataRightTo(const KeyType & x) const;
	int keyRight(const KeyType & x) const;
	int keyLeft(const KeyType & x) const;
	bool isKeyInRange(const KeyType & x) const;
	const SearchResult findKey(const KeyType & x) const;
	bool checkDupKey(KeyType & duk) const;
	void validateKeys();
	
private:
    
};

template <typename KeyType, int MaxNKey>
KeyNData<KeyType, MaxNKey>::KeyNData() :
m_numKeys(0)
{
    for(int i=0;i< MaxNKey;++i)
        m_data[i].index = NULL;
}

template <typename KeyType, int MaxNKey>
KeyNData<KeyType, MaxNKey>::~KeyNData() 
{
    for(int i=0;i< m_numKeys;++i)
		if(m_data[i].index) delete m_data[i].index;
}

template <typename KeyType, int MaxNKey>
const int & KeyNData<KeyType, MaxNKey>::numKeys() const
{ return m_numKeys; }

template <typename KeyType, int MaxNKey>
const KeyType & KeyNData<KeyType, MaxNKey>::key(const int & i) const
{ return m_data[i].key; }

template <typename KeyType, int MaxNKey>
Entity * KeyNData<KeyType, MaxNKey>::index(const int & i) const
{ return m_data[i].index; }

template <typename KeyType, int MaxNKey>
bool KeyNData<KeyType, MaxNKey>::isFull() const
{ return m_numKeys == MaxNKey; }

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::reduceNumKeys() 
{ m_numKeys--; }
	
template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::increaseNumKeys() 
{ m_numKeys++; }
	
template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::setNumKeys(int x) 
{ m_numKeys = x; }

template <typename KeyType, int MaxNKey>
bool KeyNData<KeyType, MaxNKey>::isSingular() const
{ return m_numKeys < 2; }

template <typename KeyType, int MaxNKey>
bool KeyNData<KeyType, MaxNKey>::isEmpty() const
{ return m_numKeys < 1; }

template <typename KeyType, int MaxNKey>
const Pair<KeyType, Entity> & KeyNData<KeyType, MaxNKey>::data(int x) const 
{ return m_data[x]; }

template <typename KeyType, int MaxNKey>
Pair<KeyType, Entity> * KeyNData<KeyType, MaxNKey>::dataR(int x) 
{ return &m_data[x]; }

template <typename KeyType, int MaxNKey>
const KeyType & KeyNData<KeyType, MaxNKey>::firstKey() const 
{ return m_data[0].key; }

template <typename KeyType, int MaxNKey>
const KeyType & KeyNData<KeyType, MaxNKey>::lastKey() const 
{ return m_data[numKeys() - 1].key; }

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::insertData(Pair<KeyType, Entity> x)
{	
	int i;
    for(i= numKeys() - 1;i >= 0 && m_data[i].key > x.key; i--)
        m_data[i+1] = m_data[i];
		
    m_data[i+1] = x;
	//std::cout<<"insert key "<<x.key<<" at "<<i+1<<"\n";
    increaseNumKeys();
}

template <typename KeyType, int MaxNKey>
int KeyNData<KeyType, MaxNKey>::insertKey(const KeyType & x)
{  
	if(m_numKeys < 1) {
		m_data[0].key = x;
		m_data[0].index = NULL;
		m_numKeys = 1;
		return 0;
	}
	
	int i;
    for(i= numKeys() - 1;i >= 0 && m_data[i].key > x; --i) {
		m_data[i+1] = m_data[i];
	}
		
	m_data[i+1].key = x;
	m_data[i+1].index = NULL;
	
	increaseNumKeys();
	
#if 0
	KeyType duk;
	if(!checkDupKey(duk)) {
		std::cout<<" aft insert key "<<x;
		if(duk == x) std::cout<<"\n dupleaf "<<*this;
	}
#endif
	return i+1;
}

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::setKey(int k, const KeyType & x)
{ m_data[k].key = x; }

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::setIndex(int k, Entity * x)
{ m_data[k].index = x; }

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::setData(int k, const Pair<KeyType, Entity> & x)
{ m_data[k] = x; }

template <typename KeyType, int MaxNKey> 
bool KeyNData<KeyType, MaxNKey>::hasKey(const KeyType & x) const
{
	if(isEmpty() ) return false;
    if(x > lastKey() || x < firstKey() ) return false;
	return (findKey(x).found > -1);
}

template <typename KeyType, int MaxNKey>
const Pair<KeyType, Entity> & KeyNData<KeyType, MaxNKey>::lastData() const 
{ return m_data[numKeys() - 1]; }

template <typename KeyType, int MaxNKey>
const Pair<KeyType, Entity> & KeyNData<KeyType, MaxNKey>::firstData() const 
{ return m_data[0]; }

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::removeLastData()
{
	m_data[numKeys() - 1].index = NULL;
	reduceNumKeys();
}

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::removeFirstData()
{
    if(numKeys() == 1) {
        m_data[0].index = NULL;
        reduceNumKeys();
        return;
    }
    
	for(int i = 0; i < numKeys() - 1; i++) {
		m_data[i] = m_data[i+1];
	}
	m_data[numKeys() - 1].index = NULL;
	reduceNumKeys();
}

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::replaceKey(const KeyType & x, const KeyType & y)
{
	SearchResult s = findKey(x);
	if(s.found > -1) m_data[s.found].key = y;
}

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::replaceIndex(int n, Pair<KeyType, Entity> x)
{
	m_data[n].index = x.index;
}

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::removeFirstData1()
{
    for(int i= 0; i < numKeys() - 1; i++)
		m_data[i] = m_data[i+1];
	reduceNumKeys();
}

template <typename KeyType, int MaxNKey>
bool KeyNData<KeyType, MaxNKey>::removeKeyAndData(const KeyType & x)
{
	SearchResult s = findKey(x);
	
	if(s.found < 0)
	    return false;
	
	int found = s.found;
	
	if(m_data[found].index) {
	     delete m_data[found].index;
	     m_data[found].index = 0;
	}

	if(found == numKeys() - 1) {
		reduceNumKeys();
		return true;
	}
	
	for(int i= found; i < numKeys() - 1; i++)
		m_data[i] = m_data[i+1];
		
    reduceNumKeys();
	return true;
}

template <typename KeyType, int MaxNKey>
const Pair<KeyType, Entity> KeyNData<KeyType, MaxNKey>::dataRightTo(const KeyType & x) const
{
	int i = keyRight(x);
	if(i < 0) return lastData();
	return m_data[i];
}

template <typename KeyType, int MaxNKey>
int KeyNData<KeyType, MaxNKey>::keyRight(const KeyType & x) const
{
	if(lastKey() < x) return -1;
	SearchResult s = findKey(x);
	if(s.found > -1) return s.found;
	return s.high;
}

template <typename KeyType, int MaxNKey>
int KeyNData<KeyType, MaxNKey>::keyLeft(const KeyType & x) const
{
	if(lastKey() < x) return numKeys() - 1;
	if(firstKey() >= x) return -1;
	SearchResult s = findKey(x);
	int ii = s.low;
	if(s.found > -1) ii = s.found - 1;
	else if(key(s.high) < x) ii = s.high;
	
	return ii;
}

template <typename KeyType, int MaxNKey>
bool KeyNData<KeyType, MaxNKey>::isKeyInRange(const KeyType & x) const
{
	if( x < firstKey() ) return false;
	if( x > lastKey() ) return false; 
	return true;
}

template <typename KeyType, int MaxNKey>
const SearchResult KeyNData<KeyType, MaxNKey>::findKey(const KeyType & x) const
{
	SearchResult r;
    r.found = -1;
    r.low = 0; 
	r.high = 0;

	if(numKeys() < 1) return r;
	
	r.high = numKeys() - 1;
	
	if(key(0) == x) {
		r.found = 0;
		return r;
	}
	
	if(key(r.high) == x) {
		r.found = r.high;
		return r;
	}
	
    int mid;
    while(r.low < r.high - 1) {
        mid = (r.low + r.high) / 2;
        
		if(key(mid) == x) r.found = mid;
        if(key(r.low) == x) r.found = r.low;
		if(key(r.high) == x) r.found = r.high;
		
        if(r.found > -1) break;
		
        if(x < key(mid)) r.high = mid;
        else r.low = mid;
    }
    
    return r;
}

template <typename KeyType, int MaxNKey>
bool KeyNData<KeyType, MaxNKey>::checkDupKey(KeyType & duk) const
{
	KeyType k0 = key(0);
	int i=1;
	for(;i<numKeys();++i) {
		if(key(i) > k0) 
			k0 = key(i);
		else {
			std::cout<<"\n k"<<i<<" "<<key(i)<<">="<<k0<<" ";
			duk = key(i);
			return false;
		}
	}
	return true;
}

template <typename KeyType, int MaxNKey>
void KeyNData<KeyType, MaxNKey>::validateKeys()
{	
	for(int i=0; i< numKeys()-1; ++i) {
	    if(index(i) == index(i+1)) {
	        
	        for(int j=i;j<numKeys()-1;++j) {
	            m_data[j] = m_data[j+1];
	        }
	        m_numKeys--;
	    }
	}
}

}

}

#pragma once
#include <iostream>

namespace aphid {

namespace sdb {

template <typename KeyType, int MinNKey, int MaxNKey>
class KeyNData {

    Pair<KeyType, Entity> m_data[MaxNKey];
    int m_numKeys;
    
public :
    KeyNData();
    virtual ~KeyNData();
    
protected:
    const int & numKeys() const;
	const KeyType & key(const int & i) const;
	Entity * index(const int & i) const;
    bool isFull() const;
	bool isUnderflow() const;
    void reduceNumKeys();
    void increaseNumKeys();
    void setNumKeys(int x);
    bool isSingular() const;
    bool isEmpty() const;
    const Pair<KeyType, Entity> & data(int x) const; 
    Pair<KeyType, Entity> * dataR(int x) const;

private:
    
};

template <typename KeyType, int MinNKey, int MaxNKey>
KeyNData<KeyType, MinNKey, MaxNKey>::KeyNData() 
{
    for(int i=0;i< MaxNKey;++i)
        m_data[i].index = NULL;
}

template <typename KeyType, int MinNKey, int MaxNKey>
KeyNData<KeyType, MinNKey, MaxNKey>::~KeyNData() 
{
    for(int i=0;i< MaxNKey;++i)
		if(m_data[i].index) delete m_data[i].index;
}

template <typename KeyType, int MinNKey, int MaxNKey>
const int & KeyNData<KeyType, MinNKey, MaxNKey>::numKeys() const
{ return m_numKeys; }

template <typename KeyType, int MinNKey, int MaxNKey>
const KeyType & KeyNData<KeyType, MinNKey, MaxNKey>::key(const int & i) const
{ return m_data[i].key; }

template <typename KeyType, int MinNKey, int MaxNKey>
Entity * KeyNData<KeyType, MinNKey, MaxNKey>::index(const int & i) const
{ return m_data[i].index; }

template <typename KeyType, int MinNKey, int MaxNKey>
bool KeyNData<KeyType, MinNKey, MaxNKey>::isFull() const
{ return m_numKeys == MaxNKey; }

template <typename KeyType, int MinNKey, int MaxNKey>
bool KeyNData<KeyType, MinNKey, MaxNKey>::isUnderflow() const
{ return m_numKeys < MinNKey; }

template <typename KeyType, int MinNKey, int MaxNKey>
void KeyNData<KeyType, MinNKey, MaxNKey>::reduceNumKeys() 
{ m_numKeys--; }
	
template <typename KeyType, int MinNKey, int MaxNKey>
void KeyNData<KeyType, MinNKey, MaxNKey>::increaseNumKeys() 
{ m_numKeys++; }
	
template <typename KeyType, int MinNKey, int MaxNKey>
void KeyNData<KeyType, MinNKey, MaxNKey>::setNumKeys(int x) 
{ m_numKeys = x; }

template <typename KeyType, int MinNKey, int MaxNKey>
bool KeyNData<KeyType, MinNKey, MaxNKey>::isSingular() const
{ return m_numKeys == 1; }

template <typename KeyType, int MinNKey, int MaxNKey>
bool KeyNData<KeyType, MinNKey, MaxNKey>::isEmpty() const
{ return m_numKeys < 1; }

template <typename KeyType, int MinNKey, int MaxNKey>
const Pair<KeyType, Entity> & KeyNData<KeyType, MinNKey, MaxNKey>::data(int x) const 
{ return m_data[x]; }

template <typename KeyType, int MinNKey, int MaxNKey>
Pair<KeyType, Entity> * KeyNData<KeyType, MinNKey, MaxNKey>::dataR(int x) const 
{ return &m_data[x]; }

}

}

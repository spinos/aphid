#pragma once
#include <Geometry.h>
#include <Boundary.h>
#include <KdTreeNode.h>
#include "KdSah.h"

template <int MaxLevel, typename T>
class KdGeometry : public Geometry, public Boundary
{
    /// num node 2^n + 2^n - 1
    /// num leaf node 2^n
    /// where root level is 0 leaf level is n
    KdTreeNode m_nodePool[1<< MaxLevel + 1];
    T m_leafData[1<< MaxLevel];
	int m_maxLevel;
    
public:
    KdGeometry();
	virtual ~KdGeometry();

    KdTreeNode* getRoot() const;
    KdTreeNode* root();
    int maxLevel() const;
    
protected:

private:

};

template <int MaxLevel, typename T>
KdGeometry<MaxLevel, T>::KdGeometry() 
{
    m_maxLevel = MaxLevel;
}

template <int MaxLevel, typename T>
KdGeometry<MaxLevel, T>::~KdGeometry() {}

template <int MaxLevel, typename T>
KdTreeNode * KdGeometry<MaxLevel, T>::getRoot() const
{ return &m_nodePool[0]; }

template <int MaxLevel, typename T>
KdTreeNode * KdGeometry<MaxLevel, T>::root()
{ return &m_nodePool[0]; }

template <int MaxLevel, typename T>
int KdGeometry<MaxLevel, T>::maxLevel() const
{ return m_maxLevel; }

//:~

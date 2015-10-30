#pragma once
#include <Geometry.h>
#include <Boundary.h>
#include "KdNNode.h"
#include "KdSah.h"

template <typename T, typename Tn>
class KdNTree : public Geometry, public Boundary
{
    Tn * m_nodePool;
    T  * m_leafData;
	int m_maxLevel;
    int m_maxNumNodes;
public:
    KdNTree(int maxLevel, int numPrims);
	virtual ~KdNTree();

    Tn * getRoot() const;
    Tn * root();
    Tn * getNodes() const;
    Tn * nodes();
    int maxLevel() const;
    int maxNumNodes() const;
protected:

private:

};

template <typename T, typename Tn>
KdNTree<T, Tn>::KdNTree(int maxLevel, int numPrims) 
{
    m_maxLevel = maxLevel;
    m_maxNumNodes = numPrims>>Tn::BranchingFactor();
    m_nodePool = new Tn[m_maxNumNodes];
    m_leafData = new T[numPrims<<1];
}

template <typename T, typename Tn>
KdNTree<T, Tn>::~KdNTree() 
{
    delete[] m_nodePool;
    delete[] m_leafData;
}

template <typename T, typename Tn>
Tn * KdNTree<T, Tn>::getRoot() const
{ return &m_nodePool[0]; }

template <typename T, typename Tn>
Tn * KdNTree<T, Tn>::root()
{ return &m_nodePool[0]; }

template <typename T, typename Tn>
Tn * KdNTree<T, Tn>::getNodes() const
{ return m_nodePool; }

template <typename T, typename Tn>
Tn * KdNTree<T, Tn>::nodes()
{ return m_nodePool; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::maxLevel() const
{ return m_maxLevel; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::maxNumNodes() const
{ return m_maxNumNodes; }
//:~

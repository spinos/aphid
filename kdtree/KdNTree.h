#pragma once
#include <Geometry.h>
#include <Boundary.h>
#include "KdNNode.h"
#include "KdSah.h"
#include <sstream>

template <typename T, typename Tn>
class KdNTree : public Geometry, public Boundary
{
    Tn * m_nodePool;
    T  ** m_leafData;
    int m_maxNumNodes;
	int m_maxNumData;
	int m_numNodes;
	int m_numData;
	
public:
    KdNTree(int numPrims);
	virtual ~KdNTree();

    Tn * root();
    Tn * nodes();
	
    int maxNumNodes() const;
	int maxNumData() const;
	
	int numNodes() const;
	int addNode();
	
	void addData(T * x);
	int numData() const;
	T * dataAt(int idx) const;
	
	virtual std::string verbosestr() const;
protected:

private:

};

template <typename T, typename Tn>
KdNTree<T, Tn>::KdNTree(int numPrims) 
{
    m_maxNumNodes = numPrims>>Tn::BranchingFactor-1;
	m_maxNumData = numPrims<<1;
    m_nodePool = new Tn[m_maxNumNodes];
    m_leafData = new T*[m_maxNumData];
	m_numNodes = 1;
	m_numData = 0;
}

template <typename T, typename Tn>
KdNTree<T, Tn>::~KdNTree() 
{
    delete[] m_nodePool;
    delete[] m_leafData;
}

template <typename T, typename Tn>
Tn * KdNTree<T, Tn>::root()
{ return &m_nodePool[0]; }

template <typename T, typename Tn>
Tn * KdNTree<T, Tn>::nodes()
{ return m_nodePool; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::maxNumNodes() const
{ return m_maxNumNodes; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::maxNumData() const
{ return m_maxNumData; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::numNodes() const
{ return m_numNodes; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::numData() const
{ return m_numData; }

template <typename T, typename Tn>
void KdNTree<T, Tn>::addData(T * x)
{
	m_leafData[m_numData] = x;
	m_numData++;
}

template <typename T, typename Tn>
int KdNTree<T, Tn>::addNode()
{ 
	m_numNodes++; 
	return m_numNodes - 1;
}

template <typename T, typename Tn>
T * KdNTree<T, Tn>::dataAt(int idx) const
{ return m_leafData[idx]; }

template <typename T, typename Tn>
std::string KdNTree<T, Tn>::verbosestr() const
{ 
	std::stringstream sst;
	sst<<"\n KdNTree: "
	<<"\n treelet level "<<Tn::BranchingFactor
	<<"\n n nodes(max) "<<numNodes()<<"("<<maxNumNodes()<<")"
	<<"\n n data(max)  "<<numData()<<"("<<maxNumData()<<")\n";
	return sst.str();
}
//:~

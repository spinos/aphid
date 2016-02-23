#pragma once
#include <Geometry.h>
#include <Boundary.h>
#include "KdNNode.h"
#include "KdSah.h"
#include <sstream>

namespace aphid {

class KdNeighbors {
	/// 0 left 1 right 2 bottom 3 top 4 back 5 front
public:
	BoundingBox _n[6];
	void reset() 
	{
		int i = 0;
		for(;i<6;i++) {
			_n[i].m_padding0 = 0; // node
			_n[i].m_padding1 = 0; // treelet, zero is null
		}
	}
	
	void set(const BoundingBox & box, int axis, bool isHigh, int treeletIdx, int nodeIdx)
	{
		int idx = axis<<1;
		if(isHigh) idx++;
		set(box, idx, treeletIdx, nodeIdx);
	}
	
	void set(const BoundingBox & box, int idx, int treeletIdx, int nodeIdx)
	{
		_n[idx] = box;
		_n[idx].m_padding0 = nodeIdx;
		_n[idx].m_padding1 = treeletIdx;
	}
	
	bool isEmpty() const
	{
		int i = 0;
		for(;i<6;i++) {
			if(_n[i].m_padding1 != 0) return false;
		}
		return true;
	}
	
	static bool IsNeighborOf(int dir, const BoundingBox & a, const BoundingBox & b)
	{
		const int splitAxis = dir / 2;
		int i = 0;
		for(;i<3;i++) {
			if(i==splitAxis) {
				if(dir & 1) {
					if(b.getMin(splitAxis) != a.getMax(splitAxis) ) return false;
				}
				else {
					if(b.getMax(splitAxis) != a.getMin(splitAxis) ) return false;
				}
			}
			else {
				if(b.getMin(i) > a.getMin(i)) return false;
				if(b.getMax(i) < a.getMax(i)) return false;
			}
		}
		return true;
	}
	
	unsigned encodeTreeletNodeHash(int i, int s) const
	{ return (_n[i].m_padding1 << (s + 1)) + _n[i].m_padding0; }
	
	void decodeTreeletNodeHash(int i, int s, unsigned & itreelet, unsigned & inode) const
	{
		itreelet = _n[i].m_padding1 >> s+1;
		inode = _n[i].m_padding1 & ~(1<<s+1);
	}
	
	void verbose() const
	{
		int i = 0;
		for(;i<6;i++) {
			if(_n[i].m_padding1 != 0) std::cout<<"\n ["<<i<<"] "<<_n[i].m_padding1
				<<" "<<_n[i].m_padding0
				<<" "<<_n[i];
		}
	}
};

template <typename T, typename Tn>
class KdNTree : public AVerbose, public Boundary
{
	/// i --> tree_leaf[i] --> prim_start
	///                    \-> rope_ind   --> leaf_neighbors[rope_ind]
	///
	struct TreeLeaf {
		unsigned _ropeInd[6];
		unsigned _primStart;
		unsigned _nouse;
	};
	
	sdb::VectorArray<T> * m_source;
    sdb::VectorArray<Tn> m_nodePool;
	sdb::VectorArray<TreeLeaf> m_leafNodes;
	std::vector<int> m_leafDataIndices;
	BoundingBox * m_ropes;
    unsigned m_numRopes;

public:
    KdNTree();
	virtual ~KdNTree();
	
	void init(sdb::VectorArray<T> * source, const BoundingBox & box);
	bool isNull() const;

    Tn * root();
    
	sdb::VectorArray<Tn> & nodes();
	
    int maxNumData() const;
	
	int numNodes() const;
	int addNode();
	
	void addDataIndex(int x);
	int numData() const;
	T * dataAt(unsigned idx) const;
	
	int numLeafNodes() const;
	void addLeafNode(unsigned primStart);
	const unsigned & leafPrimStart(unsigned idx) const;
	void getLeafBox(BoundingBox & dst, unsigned idx, unsigned count) const;
	
	void setLeafRope(unsigned idx, const KdNeighbors & ns);
	
	void createRopes(unsigned n);
	void setRope(unsigned idx, const BoundingBox & v );
	
	unsigned leafRopeInd(unsigned idx, int ri) const;
	void setLeafRopeInd(unsigned x, unsigned idx, int ri);

	sdb::VectorArray<T> * source();
	
	virtual std::string verbosestr() const;
    
    typedef Tn TreeletType;
protected:

private:
	void clear();
};

template <typename T, typename Tn>
KdNTree<T, Tn>::KdNTree() 
{}

template <typename T, typename Tn>
KdNTree<T, Tn>::~KdNTree() 
{
    clear();
}

template <typename T, typename Tn>
void KdNTree<T, Tn>::init(sdb::VectorArray<T> * source, const BoundingBox & box) 
{
	clear();
	Boundary::setBBox(box);
	m_source = source;
	m_numRopes = 0;
	m_nodePool.clear();
	m_nodePool.insert();
	m_leafNodes.clear();
}

template <typename T, typename Tn>
bool KdNTree<T, Tn>::isNull() const
{ return m_nodePool.size() < 1; }

template <typename T, typename Tn>
void KdNTree<T, Tn>::clear()
{
	if(isNull()) return;
	m_leafDataIndices.clear();
	if(m_ropes) delete[] m_ropes;
}

template <typename T, typename Tn>
sdb::VectorArray<T> * KdNTree<T, Tn>::source()
{ return m_source; }

template <typename T, typename Tn>
Tn * KdNTree<T, Tn>::root()
{ return m_nodePool[0]; }

template <typename T, typename Tn>
sdb::VectorArray<Tn> & KdNTree<T, Tn>::nodes()
{ return m_nodePool; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::numNodes() const
{ return m_nodePool.size(); }

template <typename T, typename Tn>
int KdNTree<T, Tn>::numData() const
{ return m_leafDataIndices.size(); }

template <typename T, typename Tn>
void KdNTree<T, Tn>::addDataIndex(int x)
{ m_leafDataIndices.push_back(x); }

template <typename T, typename Tn>
int KdNTree<T, Tn>::addNode()
{ 
	m_nodePool.insert(); 
	return m_nodePool.size() - 1;
}

template <typename T, typename Tn>
T * KdNTree<T, Tn>::dataAt(unsigned idx) const
{ return m_source->get(m_leafDataIndices[idx]); }

template <typename T, typename Tn>
int KdNTree<T, Tn>::numLeafNodes() const
{ return m_leafNodes.size(); }

template <typename T, typename Tn>
void KdNTree<T, Tn>::addLeafNode(unsigned primStart)
{ 
	TreeLeaf l;
	l._primStart = primStart;
	m_leafNodes.insert(l);
}

template <typename T, typename Tn>
const unsigned & KdNTree<T, Tn>::leafPrimStart(unsigned idx) const
{ return m_leafNodes[idx]->_primStart; }

template <typename T, typename Tn>
void KdNTree<T, Tn>::getLeafBox(BoundingBox & dst, unsigned idx, unsigned count) const
{
	dst.reset();
	const unsigned s = leafPrimStart(idx);
    unsigned i = 0;
    for(;i< count; i++) dst.expandBy( dataAt(s + i)->bbox() );
}

template <typename T, typename Tn>
void KdNTree<T, Tn>::createRopes(unsigned n)
{ 
	m_numRopes = n;
	m_ropes = new BoundingBox[n]; 
}

template <typename T, typename Tn>
void KdNTree<T, Tn>::setRope(unsigned idx, const BoundingBox & v )
{ m_ropes[idx] = v; }

template <typename T, typename Tn>
void KdNTree<T, Tn>::setLeafRope(unsigned idx, const KdNeighbors & ns)
{
	int i = 0;
	for(;i<6;i++) {
		if(ns._n[i].m_padding1 != 0) {
			m_leafNodes[idx]->_ropeInd[i] = ns.encodeTreeletNodeHash(i, Tn::BranchingFactor);
		}
		else {
			m_leafNodes[idx]->_ropeInd[i] = 0;
		}
	}
	// ns.verbose();
}

template <typename T, typename Tn>
unsigned KdNTree<T, Tn>::leafRopeInd(unsigned idx, int ri) const
{ return m_leafNodes[idx]->_ropeInd[ri]; }

template <typename T, typename Tn>
void KdNTree<T, Tn>::setLeafRopeInd(unsigned x, unsigned idx, int ri)
{ 
	// std::cout<<"\n map "<<m_leafNodes[idx]._ropeInd[ri]<<" to "<<x;
	m_leafNodes[idx]->_ropeInd[ri] = x; 
}

template <typename T, typename Tn>
std::string KdNTree<T, Tn>::verbosestr() const
{ 
	std::stringstream sst;
	sst<<"\n KdNTree: "
	<<"\n treelet level "<<Tn::BranchingFactor
	<<"\n n input "<<m_source->size()
	<<"\n n node "<<numNodes()
	<<"\n n leaf "<<numLeafNodes()
	<<"\n n data "<<numData()
	<<"\n n rope "<<m_numRopes
	<<"\n";
	return sst.str();
}

}
//:~

#pragma once
#include <kd/KdTreeNode.h>

namespace aphid {

/// http://www.highperformancegraphics.org/previous/www_2012/media/Papers/HPG2012_Papers_Heitz.pdf
/// n levels
/// 2^(n+1) - 2 nodes
///    
/// parent (last level node of another treelet)
///      /       \
///    0          1
///  2   3     4     5
/// 6 7 8 9  10 11 12 13
template <int NumLevels>
class KdNNode {
    
    KdTreeNode m_nodes[(1<<(NumLevels+1)) - 2];
	int m_nouse[4];
    
public:
	KdNNode();
	
    const KdTreeNode * node(int idx) const;
	KdTreeNode * node(int idx);
	
	void setInternal(int idx, int axis, float pos, int offset);
	void setLeaf(int idx, unsigned start, unsigned num);
	
	int internalOffset(int idx) const;
	
	void verbose() const;
	
	static int NumNodes;
	static int BranchingFactor;
	
	enum EMask {
		TreeletOffsetMask = 1<<20,
		TreeletOffsetMaskTau = ~TreeletOffsetMask
	};
	
	void printChild(const int & idx) const;

private:
    void setAllLeaf();
	
};

template <int NumLevels>
KdNNode<NumLevels>::KdNNode() 
{ setAllLeaf(); }

template <int NumLevels>
void KdNNode<NumLevels>::setAllLeaf()
{
	//for(int i=0; i<NumNodes; ++i)
	//	setLeaf(i, 0, 0);
	memset (m_nodes, 0, (NumNodes + 2) << 3);
}

template <int NumLevels>
void KdNNode<NumLevels>::setInternal(int idx, int axis, float pos, int offset)
{
/// set internal last
	m_nodes[idx].setAxis(axis);
	m_nodes[idx].setSplitPos(pos);
	m_nodes[idx].setOffset(offset);
	m_nodes[idx].setInternal();
}

template <int NumLevels>
void KdNNode<NumLevels>::setLeaf(int idx, unsigned start, unsigned num)
{
/// set leaf first
	m_nodes[idx].setLeaf();
	m_nodes[idx].setPrimStart( start); 
	m_nodes[idx].setNumPrims( num); 
}

template <int NumLevels>
void KdNNode<NumLevels>::verbose() const
{
	std::cout<<"\n treelet";
	//		<<" level "<<NumLevels
	//		<<" n node "<<NumNodes;
	int i = 0;
	for(;i<NumNodes;i++) {
		const KdTreeNode & child = m_nodes[i];
		if(child.isLeaf()) {
			if(child.getNumPrims() > 0) printChild(i);
		}
		else {
			if(child.getOffset() > 0) printChild(i);
		}
	}
}

template <int NumLevels>
void KdNNode<NumLevels>::printChild(const int & idx) const
{
	const KdTreeNode & child = m_nodes[idx];
	if(child.isLeaf()) {
		std::cout<<"\n ["<<idx<<"] leaf"
					<<" "<<child.getPrimStart()
					<<" count "<<child.getNumPrims();
	}
	else {
		std::cout<<"\n ["<<idx<<"] internal"
				<<" offset "<<(child.getOffset() & TreeletOffsetMaskTau)
				<<" axis "<<child.getAxis()
				<<" split "<<child.getSplitPos();
	}
}

template <int NumLevels>
int KdNNode<NumLevels>::internalOffset(int idx) const
{ return (node(idx)->getOffset() & TreeletOffsetMaskTau); }

template <int NumLevels>
int KdNNode<NumLevels>::NumNodes = (1<<NumLevels+1) - 2;

template <int NumLevels>
int KdNNode<NumLevels>::BranchingFactor = NumLevels;

template <int NumLevels>
const KdTreeNode * KdNNode<NumLevels>::node(int idx) const
{ return & m_nodes[idx]; }

template <int NumLevels>
KdTreeNode * KdNNode<NumLevels>::node(int idx)
{ return & m_nodes[idx]; }

typedef KdNNode<3> KdNode3;
typedef KdNNode<4> KdNode4;
typedef KdNNode<5> KdNode5;

}
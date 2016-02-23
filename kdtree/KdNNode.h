#pragma once
#include <KdTreeNode.h>

namespace aphid {

/// http://www.highperformancegraphics.org/previous/www_2012/media/Papers/HPG2012_Papers_Heitz.pdf

template <int NumLevels>
class KdNNode {
    /// n levels
    /// 2^n - 2 nodes
    KdTreeNode m_nodes[(1<<NumLevels+1) - 2];
public:
	KdNNode();
	
    KdTreeNode * node(int idx)
    { return & m_nodes[idx]; }
    
	void setInternal(int idx, int axis, float pos, int offset);
	void setLeaf(int idx, unsigned start, unsigned num);
	
	void verbose();
	
	static int NumNodes;
	static int BranchingFactor;
	
	enum EMask {
		TreeletOffsetMask = 1<<20,
	};
};

template <int NumLevels>
KdNNode<NumLevels>::KdNNode() {}

template <int NumLevels>
void KdNNode<NumLevels>::setInternal(int idx, int axis, float pos, int offset)
{
	m_nodes[idx].setAxis(axis);
	m_nodes[idx].setSplitPos(pos);
	m_nodes[idx].setOffset(offset);
	m_nodes[idx].setLeaf(false);
}

template <int NumLevels>
void KdNNode<NumLevels>::setLeaf(int idx, unsigned start, unsigned num)
{ 
	m_nodes[idx].setPrimStart( start); 
	m_nodes[idx].setNumPrims( num); 
	m_nodes[idx].setLeaf(true); 
}

template <int NumLevels>
void KdNNode<NumLevels>::verbose()
{
	std::cout<<"\n treelet level "<<NumLevels
			<<" n node "<<NumNodes;
	int i = 0;
	for(;i<NumNodes;i++) {
		std::cout<<"\n node "<<i;
		if(m_nodes[i].isLeaf()) {
			std::cout<<" leaf";
		}
		else {
			std::cout<<" internal"
					<<" split axis "<<m_nodes[i].getAxis()
					<<" pos "<<m_nodes[i].getSplitPos()
					<<" offset "<<(m_nodes[i].getOffset() & ~TreeletOffsetMask);
		}
	}
}

template <int NumLevels>
int KdNNode<NumLevels>::NumNodes = (1<<NumLevels+1) - 2;

template <int NumLevels>
int KdNNode<NumLevels>::BranchingFactor = NumLevels;

typedef KdNNode<3> KdNode3;
typedef KdNNode<4> KdNode4;

}
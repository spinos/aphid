#pragma once
#include <KdTreeNode.h>

/// http://www.highperformancegraphics.org/previous/www_2012/media/Papers/HPG2012_Papers_Heitz.pdf

template <int NumLevels>
class KdNNode {
    /// n levels
    /// 2^n - 2 nodes
    KdTreeNode m_nodes[1<<NumLevels];
public:
	KdNNode();
	
    KdTreeNode * node(int idx) const
    { return & m_nodes[idx]; }
    
	void setInternal(int idx, int axis, float pos, int offset);
	void setLeaf(int idx);
	
	void verbose();
	
	static int NumNodes();
	static int BranchingFactor();
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
void KdNNode<NumLevels>::setLeaf(int idx)
{ 
	m_nodes[idx].setLeaf(true); 
}

template <int NumLevels>
void KdNNode<NumLevels>::verbose()
{
	std::cout<<"\n treelet level "<<NumLevels
			<<" n node "<<NumNodes();
	int i = 0;
	for(;i<NumNodes();i++) {
		std::cout<<"\n node "<<i;
		if(m_nodes[i].isLeaf()) {
			std::cout<<" leaf";
		}
		else {
			std::cout<<" internal"
					<<" split axis "<<m_nodes[i].getAxis()
					<<" pos "<<m_nodes[i].getSplitPos()
					<<" offset "<<m_nodes[i].getOffset();
		}
	}
}

template <int NumLevels>
int KdNNode<NumLevels>::NumNodes() 
{ return (1<<NumLevels+1) - 2; }

template <int NumLevels>
int KdNNode<NumLevels>::BranchingFactor()
{ return NumLevels; }

typedef KdNNode<3> KdNode3;
typedef KdNNode<4> KdNode4;

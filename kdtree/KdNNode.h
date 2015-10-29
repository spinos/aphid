#pragma once
#include <KdTreeNode.h>

/// http://www.highperformancegraphics.org/previous/www_2012/media/Papers/HPG2012_Papers_Heitz.pdf

template <int NumLevels>
class KdNNode {
    /// n levels
    /// 2^n - 1 nodes
    KdTreeNode m_nodes[1<<NumLevels];
public:
	static int NumNodes;
    static int NumSplits() 
    { return NumNodes; }
    
    KdTreeNode * node(int idx) const
    { return & m_nodes[idx]; }
    
};

template <int NumLevels>
int KdNNode<NumLevels>::NumNodes = (1<<NumLevels) - 1;

typedef KdNNode<3> KdNode3;

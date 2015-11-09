#pragma once

#include <ViewFrame.h>
#include <GjkIntersection.h>
#include <KdTreeNode.h>

template<typename T>
class KdScreen {
	ViewFrame m_subFrames[1<<12];
	ViewFrame m_base;
	unsigned char * m_rgba;
	float * m_z;
	std::vector<unsigned > m_boxes;
    
public:
    KdScreen();
	virtual ~KdScreen();
	
	void create(int w, int h);
	void setView(const Frustum & f);
    bool getVisibleFrames(T * tree);
    
    unsigned numBoxes() const;
    unsigned box(unsigned idx) const;
private:
    void clear();
    void splitInterialNode(T * tree, int iTreelet, int iNode, const BoundingBox & box,
                           const Frustum & fru);
    bool firstLeafNode(unsigned & dst, T * tree, int iTreelet, int iNode, const BoundingBox & box,
                       const Frustum & fru);
    void addToBoxes(T * tree, unsigned iLeafNode, unsigned count);
};

template<typename T>
KdScreen<T>::KdScreen() 
{
    m_rgba = NULL;
    m_z = NULL;
}

template<typename T>
KdScreen<T>::~KdScreen() 
{
    clear();
}

template<typename T>
void KdScreen<T>::clear()
{
    if(m_rgba) delete[] m_rgba;
    if(m_z) delete[] m_z;
}

template<typename T>
void KdScreen<T>::create(int w, int h)
{
    clear();
    m_rgba = new unsigned char[w * h * 4];
    m_z = new float[w * h];
    
    m_base.setRect(0, 0, w-1, h-1);
}

template<typename T>
void KdScreen<T>::setView(const Frustum & f)
{
    m_base.setView(f);
}

template<typename T>
bool KdScreen<T>::getVisibleFrames(T * tree)
{
    m_boxes.clear();
    
    const BoundingBox box = tree->getBBox();
    if(!gjk::Intersect1<Frustum, BoundingBox >::Evaluate(m_base.view(), box )) 
        return false;
    
    splitInterialNode(tree, 0, 0, box, m_base.view() );
    
    return true;
}

template<typename T>
void KdScreen<T>::splitInterialNode(T * tree, int iTreelet, int iNode, const BoundingBox & box,
                                    const Frustum & fru)
{
#if 0
    unsigned iLeaf; 
    if( firstLeafNode(iLeaf, tree, iTreelet, iNode, box,
                                fru) ) return;
#else
    KdTreeNode * node = tree->nodes()[iTreelet].node(iNode);
    if(node->isLeaf()) {
        addToBoxes(tree, node->getPrimStart(), node->getNumPrims());
        return;
    }
    
    const int axis = node->getAxis();
	const float pos = node->getSplitPos();
    BoundingBox lft, rgt;
	box.split(axis, pos, lft, rgt);
    
    const bool hasLft = gjk::Intersect1<Frustum, BoundingBox >::Evaluate(fru, lft );
    const bool hasRgt = gjk::Intersect1<Frustum, BoundingBox >::Evaluate(fru, rgt );
    
    int offset = node->getOffset();
    if(offset > T::TreeletType::TreeletOffsetMask ) {
        offset &= ~T::TreeletType::TreeletOffsetMask;
        if(hasLft) {
            splitInterialNode(tree, iTreelet + offset, 0, lft, fru );
        }
        if(hasRgt) {
            splitInterialNode(tree, iTreelet + offset, 1, rgt, fru );
        }
    }
    else {
        splitInterialNode(tree, iTreelet, iNode + offset, lft, fru);
		splitInterialNode(tree, iTreelet, iNode + offset + 1, rgt, fru);
    }
#endif
}

template<typename T>
bool KdScreen<T>::firstLeafNode(unsigned & dst, T * tree, int iTreelet, int iNode, const BoundingBox & box,
                                const Frustum & fru)
{
    KdTreeNode * node = tree->nodes()[iTreelet].node(iNode);
    if(node->isLeaf()) {
        if(node->getNumPrims() < 1) return false;
        dst = node->getPrimStart();
        addToBoxes(tree, node->getPrimStart(), node->getNumPrims());
        return true;
    }
    
    const int axis = node->getAxis();
	const float pos = node->getSplitPos();
    BoundingBox lft, rgt;
	box.split(axis, pos, lft, rgt);
    
    bool isFar = false;
    int offset = node->getOffset();
    if(offset > T::TreeletType::TreeletOffsetMask ) {
        offset &= ~T::TreeletType::TreeletOffsetMask;
        isFar = true;
    }
    
    bool hasLeaf = false;
    if(gjk::Intersect1<Frustum, BoundingBox >::Evaluate(fru, lft )) {
        if(isFar) 
            hasLeaf = firstLeafNode(dst, tree, iTreelet + offset, 0, lft, fru );
        else 
            hasLeaf = firstLeafNode(dst, tree, iTreelet, iNode + offset, lft, fru);   
    }
    
    if(hasLeaf) return true;

    if(gjk::Intersect1<Frustum, BoundingBox >::Evaluate(fru, rgt )) {
        if(isFar) 
            hasLeaf = firstLeafNode(dst, tree, iTreelet + offset, 1, rgt, fru );
        else 
            hasLeaf = firstLeafNode(dst, tree, iTreelet, iNode + offset + 1, rgt, fru);
    }

    return hasLeaf;
}

template<typename T>
void KdScreen<T>::addToBoxes(T * tree, unsigned iLeafNode, unsigned count)
{
    const unsigned s = tree->leafPrimStart(iLeafNode );
    unsigned i = 0;
    for(;i< count; i++) m_boxes.push_back(i+s);
}

template<typename T>
unsigned KdScreen<T>::numBoxes() const
{ return m_boxes.size(); }

template<typename T>
unsigned KdScreen<T>::box(unsigned idx) const
{ return m_boxes[idx]; }
//:~

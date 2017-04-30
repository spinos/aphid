#pragma once

#include <ViewFrame.h>
#include <GjkIntersection.h>
#include <KdTreeNode.h>

namespace aphid {

template<typename T>
class KdScreen {
	ViewFrame m_base;
	RectangleI * m_tiles;
	unsigned char * m_rgba;
	float * m_z;
	std::vector<unsigned > m_boxes;
	int m_numTiles;
    
public:
    KdScreen();
	virtual ~KdScreen();
	
	void create(int w, int h);
	void setView(const Frustum & f);
    bool getVisibleFrames(T * tree);
    
    unsigned numBoxes() const;
    unsigned box(unsigned idx) const;
	
	std::vector<ViewFrame > m_views;
private:
    void clear();
    bool splitInterialNode(T * tree, int iTreelet, int iNode, const BoundingBox & box,
                           const ViewFrame & frm);
	bool splitLeafNode(KdTreeNode * node, const BoundingBox & box,
                           const ViewFrame & frm);
						   
    bool firstLeafNode(BoundingBox & dst, T * tree, int iTreelet, int iNode, const BoundingBox & box,
                       const Frustum & fru);
    void addToBoxes(T * tree, unsigned iLeafNode, unsigned count);
	void createTiles();
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
	if(m_tiles) delete[] m_tiles;
}

template<typename T>
void KdScreen<T>::create(int w, int h)
{
    clear();
    m_rgba = new unsigned char[w * h * 4];
    m_z = new float[w * h];
    m_base.setRect(0, 0, w-1, h-1);
	
	createTiles();
}

template<typename T>
void KdScreen<T>::createTiles()
{
	int ns = m_base.rect().width() >> 5;
	if(ns & 31) ns++;
	int nt = m_base.rect().height() >> 5;
	if(nt & 31) nt++;
	
	const int n = (ns * nt);
	m_tiles = new RectangleI[n];
	std::cout<<"\n max n tiles "<<n;
	m_numTiles = n;
	
	int i, j;
	for(j=0;j<nt;j++) {
		for(i=0;i<ns;i++) {
			
		}
	}
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
    m_views.clear();
    const BoundingBox box = tree->getBBox();
    // splitInterialNode(tree, 0, 0, box, m_base );
    
    return true;
}

template<typename T>
bool KdScreen<T>::splitInterialNode(T * tree, int iTreelet, int iNode, const BoundingBox & box,
                                    const ViewFrame & frm)
{
#if 1	
	if(frm.numPixels() <= 1024) {
		m_views.push_back(frm);
		return true;
	}

	ViewFrame child0, child1;
	frm.split(child0, child1);
	
	BoundingBox tbox;
	if(firstLeafNode(tbox, tree, iTreelet, iNode, box,
                                child0.view()) )
			splitInterialNode(tree, iTreelet, iNode, box, child0 );
			
	if(firstLeafNode(tbox, tree, iTreelet, iNode, box,
                                child1.view()) )
			splitInterialNode(tree, iTreelet, iNode, box, child1 );
/*
	if(gjk::Intersect1<Frustum, BoundingBox >::Evaluate(child0.view(), lftBox ) ) {
		if(isFar)
			splitInterialNode(tree, iTreelet + offset, 0, lftBox, child0 );
		else
			splitInterialNode(tree, iTreelet, iNode + offset, lftBox, child0); 
	}
	
	if(gjk::Intersect1<Frustum, BoundingBox >::Evaluate(child1.view(), lftBox ) ) {
		if(isFar)
			splitInterialNode(tree, iTreelet + offset, 0, lftBox, child1 );
		else
			splitInterialNode(tree, iTreelet, iNode + offset, lftBox, child1); 
	}
	
	if(gjk::Intersect1<Frustum, BoundingBox >::Evaluate(child0.view(), rgtBox ) ) {
		if(isFar)
			splitInterialNode(tree, iTreelet + offset, 1, rgtBox, child0 );
		else
			splitInterialNode(tree, iTreelet, iNode + offset + 1, rgtBox, child0); 
	}
	
	if(gjk::Intersect1<Frustum, BoundingBox >::Evaluate(child1.view(), rgtBox ) ) {
		if(isFar)
			splitInterialNode(tree, iTreelet + offset, 1, rgtBox, child1 );
		else
			splitInterialNode(tree, iTreelet, iNode + offset + 1, rgtBox, child1); 
	}
*/	
	return false;
	
#else
    
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
bool KdScreen<T>::splitLeafNode(KdTreeNode * node, const BoundingBox & box,
                           const ViewFrame & frm)
{
	if(node->getNumPrims() < 1) return false;
	
	if(frm.numPixels() <= 1024) {
		m_views.push_back(frm);
		return true;
	}
	
	ViewFrame child0, child1;
	frm.split(child0, child1);

	if(gjk::Intersect1<Frustum, BoundingBox >::Evaluate(child0.view(), box ) ) {
		// std::cout<<"\n lft frm "<<child0;
		splitLeafNode(node, box, child0 );
	}
	
	if(gjk::Intersect1<Frustum, BoundingBox >::Evaluate(child1.view(), box ) ) {
		// std::cout<<"\n rgt frm "<<child1;
		splitLeafNode(node, box, child1 );
	}
	
	return true;
}

template<typename T>
bool KdScreen<T>::firstLeafNode(BoundingBox & dst, T * tree, int iTreelet, int iNode, const BoundingBox & box,
                                const Frustum & fru)
{
    KdTreeNode * node = tree->nodes()[iTreelet].node(iNode);
    if(node->isLeaf()) {
        if(node->getNumPrims() < 1) return false;
        tree->getLeafBox(dst, node->getPrimStart(), node->getNumPrims() );
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

}
//:~

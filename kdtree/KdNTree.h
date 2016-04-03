#pragma once
#include <Geometry.h>
#include <Boundary.h>
#include <TreeProperty.h>
#include <IntersectionContext.h>
#include "KdNNode.h"
#include "KdSah.h"
#include "BoxNeighbors.h"
#include <sstream>

namespace aphid {

namespace knt {
/// i --> tree_leaf[i] --> prim_start
///					   --> prim_len
///                    --> rope_ind   --> leaf_neighbors[rope_ind]
///
struct TreeLeaf {
/// -x +x -y +y -z +z
/// start by 0, -1 is null
	int _ropeInd[6];
	int _primStart;
	int _primLength;
};
}

template <typename T, typename Tn>
class KdNTree : public AVerbose, public Boundary, public TreeProperty
{
	sdb::VectorArray<T> * m_source;
    sdb::VectorArray<Tn> m_nodePool;
	sdb::VectorArray<knt::TreeLeaf> m_leafNodes;
	sdb::VectorArray<int> m_primIndices;
	sdb::VectorArray<BoundingBox> m_ropes;
    
public:
    KdNTree();
	virtual ~KdNTree();
	
	void init(sdb::VectorArray<T> * source, const BoundingBox & box);
	bool isEmpty() const;
	int numBranches() const;
	int numLeafNodes() const;
	int numPrimIndirection() const;
	int numRopes() const;
	
	void setRelativeTransform(const BoundingBox & rel);
/// translate and scale
	void getRelativeTransform(float * dst) const;
	
    Tn * root();
	const Tn * root() const;
    const sdb::VectorArray<Tn> & branches() const;
	sdb::VectorArray<Tn> & branches();
	const sdb::VectorArray<knt::TreeLeaf> & leafNodes() const;
	const sdb::VectorArray<BoundingBox> & ropes() const;
	const sdb::VectorArray<int> & primIndirection() const;
	
	const int & primIndirectionAt(const int & idx) const;
	int addBranch();
	
	void addDataIndex(int x);
	T * dataAt(unsigned idx) const;
	
	void addLeafNode(const int & primStart, const int & primLen);
	const int & leafPrimStart(unsigned idx) const;
	void leafPrimStartLength(int & start, int & len, 
									unsigned idx) const;
	void setLeafRope(unsigned idx, const BoxNeighbors & ns);
	
	BoundingBox * addRope();
	
	const int & leafRopeInd(unsigned idx, int ri) const;
	void setLeafRopeInd(unsigned x, unsigned idx, int ri);

	const sdb::VectorArray<T> * source() const;
	virtual void setSource(sdb::VectorArray<T> * src);
	
	char intersect(IntersectionContext * ctx);
	char intersectBox(BoxIntersectContext * ctx);
	
	const T * getSource(const int x) const;
	
	virtual std::string verbosestr() const;
    
    typedef Tn TreeletType;
	
protected:
	void clear(const BoundingBox & b);
	Tn * addTreelet();
	knt::TreeLeaf * addLeaf();
	int * addIndirection();

private:
	void clear();
	int firstVisit(IntersectionContext * ctx,
							const KdTreeNode * n) const;
/// -1: non-leaf
///  0: empty leaf
///  1: non-empty leaf
	int visitLeaf(IntersectionContext * ctx, 
							int & branchIdx,
							int & nodeIdx,
							const KdTreeNode * r);
	int hitPrimitive(IntersectionContext * ctx, 
							const KdTreeNode * r);
	bool climbRope(IntersectionContext * ctx, 
							int & branchIdx,
							int & nodeIdx,
							const KdTreeNode * r);
							
	void leafIntersectBox(BoxIntersectContext * ctx,
							KdTreeNode * r);
	void innerIntersectBox(BoxIntersectContext * ctx,
							int branchIdx,
							int nodeIdx,
							const BoundingBox & b);
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
	clear(box);
/// node[0]
	m_nodePool.insert();
	m_source = source;
}

template <typename T, typename Tn>
bool KdNTree<T, Tn>::isEmpty() const
{ return m_nodePool.size() < 1; }

template <typename T, typename Tn>
void KdNTree<T, Tn>::clear()
{
	if(isEmpty()) return;
	m_nodePool.clear();
	m_leafNodes.clear();
	m_primIndices.clear();
	m_ropes.clear();
}

template <typename T, typename Tn>
void KdNTree<T, Tn>::clear(const BoundingBox & b)
{
	clear();
	Boundary::setBBox(b);
/// first rope is bound
	*addRope() = b;
	resetPropery();
	setTotalVolume(b.volume() );
}

template <typename T, typename Tn>
const sdb::VectorArray<T> * KdNTree<T, Tn>::source() const
{ return m_source; }

template <typename T, typename Tn>
Tn * KdNTree<T, Tn>::root()
{ return m_nodePool[0]; }

template <typename T, typename Tn>
const Tn * KdNTree<T, Tn>::root() const
{ return m_nodePool[0]; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::numBranches() const
{ return m_nodePool.size(); }

template <typename T, typename Tn>
void KdNTree<T, Tn>::addDataIndex(int x)
{ m_primIndices.insert(x); }

template <typename T, typename Tn>
int KdNTree<T, Tn>::addBranch()
{ 
	m_nodePool.insert(); 
	return m_nodePool.size() - 1;
}

template <typename T, typename Tn>
Tn * KdNTree<T, Tn>::addTreelet()
{ 
	m_nodePool.insert(); 
	return m_nodePool.last();
}

template <typename T, typename Tn>
T * KdNTree<T, Tn>::dataAt(unsigned idx) const
{ return m_source->get(*m_primIndices[idx]); }

template <typename T, typename Tn>
int KdNTree<T, Tn>::numLeafNodes() const
{ return m_leafNodes.size(); }

template <typename T, typename Tn>
void KdNTree<T, Tn>::addLeafNode(const int & primStart, const int & primLen)
{ 
	knt::TreeLeaf l;
	l._primStart = primStart;
	l._primLength = primLen;
	m_leafNodes.insert(l);
}

template <typename T, typename Tn>
knt::TreeLeaf * KdNTree<T, Tn>::addLeaf()
{ 
	m_leafNodes.insert();
	return m_leafNodes.last();
}

template <typename T, typename Tn>
int * KdNTree<T, Tn>::addIndirection()
{
	m_primIndices.insert();
	return m_primIndices.last();
}

template <typename T, typename Tn>
BoundingBox * KdNTree<T, Tn>::addRope()
{
	m_ropes.insert();
	return m_ropes.last();
}

template <typename T, typename Tn>
const int & KdNTree<T, Tn>::leafPrimStart(unsigned idx) const
{ return m_leafNodes[idx]->_primStart; }

template <typename T, typename Tn>
void KdNTree<T, Tn>::leafPrimStartLength(int & start, int & len, 
									unsigned idx) const
{ 
	start = m_leafNodes[idx]->_primStart; 
	len = m_leafNodes[idx]->_primLength;
}

template <typename T, typename Tn>
void KdNTree<T, Tn>::setLeafRope(unsigned idx, const BoxNeighbors & ns)
{
	// ns.verbose();
	int i = 0;
	for(;i<6;i++) {
		if(ns._n[i].m_padding1 > 0) {
			m_leafNodes[idx]->_ropeInd[i] = ns.encodeTreeletNodeHash(i, Tn::BranchingFactor);
		}
		else {
			m_leafNodes[idx]->_ropeInd[i] = -1;
		}
	}
}

template <typename T, typename Tn>
const int & KdNTree<T, Tn>::leafRopeInd(unsigned idx, int ri) const
{ return m_leafNodes[idx]->_ropeInd[ri]; }

template <typename T, typename Tn>
void KdNTree<T, Tn>::setLeafRopeInd(unsigned x, unsigned idx, int ri)
{ 
	// if(x<1) std::cout<<"\n warning map leaf "<<idx<<" rope "<<ri<<" to "<<x;
	m_leafNodes[idx]->_ropeInd[ri] = x; 
}

template <typename T, typename Tn>
const sdb::VectorArray<knt::TreeLeaf> & KdNTree<T, Tn>::leafNodes() const
{ return m_leafNodes; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::numRopes() const
{ return m_ropes.size(); }

template <typename T, typename Tn>
const sdb::VectorArray<BoundingBox> & KdNTree<T, Tn>::ropes() const
{ return m_ropes; }

template <typename T, typename Tn>
const sdb::VectorArray<int> & KdNTree<T, Tn>::primIndirection() const
{ return m_primIndices; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::numPrimIndirection() const
{ return m_primIndices.size(); }

template <typename T, typename Tn>
const int & KdNTree<T, Tn>::primIndirectionAt(const int & idx) const
{ return *m_primIndices[idx]; }

template <typename T, typename Tn>
void KdNTree<T, Tn>::setSource(sdb::VectorArray<T> * src)
{ m_source = src; }

template <typename T, typename Tn>
int KdNTree<T, Tn>::firstVisit(IntersectionContext * ctx, 
								const KdTreeNode * n) const
{
	const int axis = n->getAxis();
	const float splitPos = n->getSplitPos();
	
	const Ray & ray = ctx->m_ray;
	
	const float o = ray.m_origin.comp(axis);
	const float d = ray.m_dir.comp(axis);
	
	const BoundingBox & b = ctx->getBBox();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	// std::cout<<"\n split "<<axis<<" "<<splitPos;
	
	bool above = o >= splitPos;
/// parallel to split
	if(Absolute<float>(d) < 1e-5f) {
		if(above) ctx->setBBox(rgtBox);
		else ctx->setBBox(lftBox);
		
		return above;
	}
	
/// inside
	if(lftBox.isPointInside(ray.m_origin) ) {
		ctx->setBBox(lftBox);
		return 0;
	}
	
	if(rgtBox.isPointInside(ray.m_origin) ) {
		ctx->setBBox(rgtBox);
		return 1;
	}
	
/// one side
	float t = (splitPos - o) / d;
	if(t< 0.f) {
		if(above) ctx->setBBox(rgtBox);
		else ctx->setBBox(lftBox);
		return above;
	}
	
/// near one
	Vector3F hitP = ray.travel(t);
	if(b.isPointInside(hitP) ) {
		if(above) ctx->setBBox(rgtBox);
		else ctx->setBBox(lftBox);
		return above;
	}
	
	BoundingBox p2h;
	p2h.expandBy(ray.m_origin);
	p2h.expandBy(hitP);
/// swap side
	if(!p2h.touch(b) ) above = !above;
	
	if(above) ctx->setBBox(rgtBox);
	else ctx->setBBox(lftBox);
		
	return above;
}

template <typename T, typename Tn>
char KdNTree<T, Tn>::intersect(IntersectionContext * ctx)
{
	if(isEmpty()) return 0;
	
	const BoundingBox & b = getBBox();
	if(!b.intersect(ctx->m_ray)) return 0;
	
	ctx->setBBox(b);
	
	KdTreeNode * r = root()->node(0);
	if(r->isLeaf() ) {
		ctx->m_leafIdx = r->getPrimStart();
		hitPrimitive(ctx, r);
		return ctx->m_success;
	}
	
	int branchIdx = root()->internalOffset(0);
	int preBranchIdx = branchIdx;
	Tn * currentBranch = branches()[branchIdx];
	int nodeIdx = firstVisit(ctx, r);
	KdTreeNode * kn = currentBranch->node(nodeIdx);
	int stat;
	bool hasNext = true;
	while (hasNext) {
		stat = visitLeaf(ctx, branchIdx, nodeIdx, 
							kn);
							
		if(preBranchIdx != branchIdx) {
			currentBranch = branches()[branchIdx];
			preBranchIdx = branchIdx;
		}
		
		kn = currentBranch->node(nodeIdx);
		
		if(stat > 0 ) {
			if(hitPrimitive(ctx, kn ) ) hasNext = false;
			else stat = 0;
		}
		if(stat==0) {
			hasNext = climbRope(ctx, branchIdx, nodeIdx, 
							kn );
							
			if(preBranchIdx != branchIdx) {
				currentBranch = branches()[branchIdx];
				preBranchIdx = branchIdx;
			}
			
			kn = currentBranch->node(nodeIdx);
		}
	}
	return ctx->m_success;
}

template <typename T, typename Tn>
int KdNTree<T, Tn>::hitPrimitive(IntersectionContext * ctx, 
									const KdTreeNode * r)
{
	int start, len;
	leafPrimStartLength(start, len, r->getPrimStart() );
	std::cout<<"\n n prim "<<len;
	int nhit = 0;
	int i = 0;
	for(;i<len;++i) {
		const T * c = m_source->get(primIndirectionAt(start + i));
		if(c->intersect(ctx->m_ray, &ctx->m_tmin, &ctx->m_tmax) ) {
			ctx->m_hitP = ctx->m_ray.travel(ctx->m_tmin);
			ctx->m_ray.m_tmax = ctx->m_tmin;
			ctx->m_success = 1;
/// ind to source
			ctx->m_componentIdx = start + i;
			nhit++;
		}
	}
	std::cout<<"\n hit "<<nhit<<"\n";
	std::cout.flush();
	return nhit;
}

template <typename T, typename Tn>
int KdNTree<T, Tn>::visitLeaf(IntersectionContext * ctx, 
									int & branchIdx,
									int & nodeIdx,
									const KdTreeNode * r)
{
	std::cout<<"\n node "<<nodeIdx;
				
	if(r->isLeaf() ) {
		std::cout<<"\n hit leaf "<<r->getPrimStart();
		ctx->m_leafIdx = r->getPrimStart();
		if(r->getNumPrims() < 1) {
			return 0;
		}
/// leaf ind actually
		ctx->m_componentIdx = r->getPrimStart();
		return 1;
	}
	
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		// std::cout<<"\n inner offset "<<offset;
		nodeIdx += offset + firstVisit(ctx, r);
	}
	else {
		branchIdx += offset & Tn::TreeletOffsetMaskTau;
		std::cout<<"\n branch "<<branchIdx;
		nodeIdx = firstVisit(ctx, r);
	}
	
	return -1;
}

template <typename T, typename Tn>
bool KdNTree<T, Tn>::climbRope(IntersectionContext * ctx, 
									int & branchIdx,
									int & nodeIdx,
									const KdTreeNode * r)
{
	const BoundingBox & b = ctx->getBBox();
	float t0, t1;
	b.intersect(ctx->m_ray, &t0, &t1);
	Vector3F hit1 = ctx->m_ray.travel(t1 + 1e-3f);
	if(b.isPointInside(hit1) ) {
		std::cout<<"\n end "<<hit1
			<<"\n inside box "<<b;
			return false;
	}
	
	int side = b.pointOnSide(hit1);
	std::cout<<"\n ray "<<ctx->m_ray.m_origin<<" "<<ctx->m_ray.m_dir
			<<"\n box "<<b
			<<"\n hit "<<hit1
			<<"\n rope side "<<side;
	
/// leaf ind actually 
	int iLeaf = r->getPrimStart();
	int iRope = leafRopeInd(iLeaf, side);
	
	if(iRope < 1) {
		std::cout<<" no rope";
		return false;
	}
	
	std::cout<<" rope["<<iRope<<"]";
	const BoundingBox * rp = m_ropes[ iRope ];
	BoxNeighbors::DecodeTreeletNodeHash(rp->m_padding1, Tn::BranchingFactor, 
					branchIdx, nodeIdx);
	std::cout<<"\n branch "<<branchIdx;
	ctx->setBBox(*rp);
	return true;
}

template <typename T, typename Tn>
const sdb::VectorArray<Tn> & KdNTree<T, Tn>::branches() const
{ return m_nodePool; }

template <typename T, typename Tn>
sdb::VectorArray<Tn> & KdNTree<T, Tn>::branches()
{ return m_nodePool; }

template <typename T, typename Tn>
std::string KdNTree<T, Tn>::verbosestr() const
{ 
	std::stringstream sst;
	sst<<"\n KdNTree: "
	<<"\n treelet level "<<Tn::BranchingFactor
	<<"\n n input "<<m_source->size()
	<<"\n n treelet "<<numBranches()
	<<"\n n leaf "<<numLeafNodes()
	<<"\n n data "<<numPrimIndirection()
	<<"\n n rope "<<numRopes()
	<<"\n";
	sst<<logProperty();
	return sst.str();
}

template <typename T, typename Tn>
char KdNTree<T, Tn>::intersectBox(BoxIntersectContext * ctx)
{
	if(isEmpty()) return 0;
	
	const BoundingBox & b = getBBox();
	if(!b.intersect(*ctx)) return 0;
	
	KdTreeNode * r = root()->node(0);
	if(r->isLeaf() ) {
		leafIntersectBox(ctx, r);
	}
	else {
		const int axis = r->getAxis();
		const float splitPos = r->getSplitPos();
		BoundingBox lftBox, rgtBox;
		b.split(axis, splitPos, lftBox, rgtBox);
		
		int branchIdx = root()->internalOffset(0);
		innerIntersectBox(ctx, branchIdx, 0, lftBox);
		innerIntersectBox(ctx, branchIdx, 1, rgtBox);
	} 
	
	return ctx->numIntersect() > 0;
}


template <typename T, typename Tn>
void KdNTree<T, Tn>::leafIntersectBox(BoxIntersectContext * ctx,
							KdTreeNode * r)
{
	if(r->getNumPrims() < 1) return;
	int start, len;
	leafPrimStartLength(start, len, r->getPrimStart() );
	int i = 0;
	for(;i<len;++i) {
		const T * c = m_source->get(primIndirectionAt(start + i) );
		if(c->calculateBBox().intersect(*ctx) ) {
			if(ctx->isExact() ) {
				if(c-> template exactIntersect<BoxIntersectContext >(*ctx) )
					ctx->addPrim(primIndirectionAt(start + i) );
			}
			else
				ctx->addPrim(primIndirectionAt(start + i) );
				
			if(ctx->isFull() ) return;
		}
	}
}

template <typename T, typename Tn>
void KdNTree<T, Tn>::innerIntersectBox(BoxIntersectContext * ctx,
							int branchIdx,
							int nodeIdx,
							const BoundingBox & b)
{
	Tn * currentBranch = branches()[branchIdx];
	KdTreeNode * r = currentBranch->node(nodeIdx);
	if(r->isLeaf() ) {
		leafIntersectBox(ctx, r);
		return;
	}
	
	const int axis = r->getAxis();
	const float splitPos = r->getSplitPos();
	BoundingBox lftBox, rgtBox;
	b.split(axis, splitPos, lftBox, rgtBox);
	
	const int offset = r->getOffset();
	if(offset < Tn::TreeletOffsetMask) {
		if(ctx->getMin(axis) < splitPos ) {
			innerIntersectBox(ctx, 
							branchIdx,
							nodeIdx + offset,
							lftBox);
			if(ctx->isFull() ) return;
		}
		
		if(ctx->getMax(axis) > splitPos ) {
			innerIntersectBox(ctx, 
							branchIdx,
							nodeIdx + offset + 1,
							rgtBox);
			if(ctx->isFull() ) return;
		}
	}
	else {
		if(ctx->getMin(axis) < splitPos ) {
			innerIntersectBox(ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							0,
							lftBox);
			if(ctx->isFull() ) return;
		}
		
		if(ctx->getMax(axis) > splitPos ) {
			innerIntersectBox(ctx, 
							branchIdx + offset & Tn::TreeletOffsetMaskTau,
							1,
							rgtBox);
			if(ctx->isFull() ) return;
		}
	}
}

template <typename T, typename Tn>
const T * KdNTree<T, Tn>::getSource(const int x) const
{ return m_source->get(primIndirectionAt(x) ); }

template <typename T, typename Tn>
void KdNTree<T, Tn>::setRelativeTransform(const BoundingBox & rel)
{
/// use enpty node of root branch to store translate and scale
	float * ts = (float *)root()->node(1);
	ts[0] = rel.getMin(0);
	ts[1] = rel.getMin(1);
	ts[2] = rel.getMin(2);
	ts[3] = rel.distance(0) / getBBox().distance(0);
	ts[4] = rel.distance(1) / getBBox().distance(1);
	ts[5] = rel.distance(2) / getBBox().distance(2);
}

template <typename T, typename Tn>
void KdNTree<T, Tn>::getRelativeTransform(float * dst) const
{
	float * ts = (float *)root()->node(1);
	memcpy(dst, ts, 24);
}

}
//:~

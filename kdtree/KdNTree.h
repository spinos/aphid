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
	BoundingBox * m_ropes;
    int m_numRopes;

public:
    KdNTree();
	virtual ~KdNTree();
	
	void init(sdb::VectorArray<T> * source, const BoundingBox & box);
	bool isEmpty() const;

    Tn * root();
    
	int numPrimIndirection() const;
	const int & primIndirectionAt(const int & idx) const;
	int numBranches() const;
	int addBranch();
	
	void addDataIndex(int x);
	T * dataAt(unsigned idx) const;
	const sdb::VectorArray<Tn> & nodes() const;
	sdb::VectorArray<Tn> & nodes();

	const sdb::VectorArray<knt::TreeLeaf> & leafNodes() const;
	
	int numLeafNodes() const;
	void addLeafNode(const int & primStart, const int & primLen);
	const int & leafPrimStart(unsigned idx) const;
	void leafPrimStartLength(int & start, int & len, 
									unsigned idx) const;
	
	void setLeafRope(unsigned idx, const BoxNeighbors & ns);
	
	void createRopes(unsigned n);
	void setRope(unsigned idx, const BoundingBox & v );
	const int & numRopes() const;
	
	const int & leafRopeInd(unsigned idx, int ri) const;
	void setLeafRopeInd(unsigned x, unsigned idx, int ri);

	sdb::VectorArray<T> * source();
	void setSource(sdb::VectorArray<T> * src);
	
	char intersect(IntersectionContext * ctx);
	
	virtual std::string verbosestr() const;
    
    typedef Tn TreeletType;
	
protected:
	const BoundingBox * ropes() const;
	BoundingBox * ropesR(const int & idx);
	const sdb::VectorArray<int> & primIndirection() const;
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
							int & nodeIdx);
	bool climbRope(IntersectionContext * ctx, 
							int & branchIdx,
							int & nodeIdx);
};

template <typename T, typename Tn>
KdNTree<T, Tn>::KdNTree() :
m_ropes(NULL),
m_numRopes(0)
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
	if(m_ropes) delete[] m_ropes;
	m_numRopes = 0;
}

template <typename T, typename Tn>
void KdNTree<T, Tn>::clear(const BoundingBox & b)
{
	clear();
	Boundary::setBBox(b);
	resetPropery();
	setTotalVolume(b.volume() );
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
const sdb::VectorArray<Tn> & KdNTree<T, Tn>::nodes() const
{ return m_nodePool; }

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
void KdNTree<T, Tn>::createRopes(unsigned n)
{ 
	m_numRopes = n;
	m_ropes = new BoundingBox[n]; 
}

template <typename T, typename Tn>
void KdNTree<T, Tn>::setRope(unsigned idx, const BoundingBox & v )
{ m_ropes[idx] = v; }

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
const int & KdNTree<T, Tn>::numRopes() const
{ return m_numRopes; }

template <typename T, typename Tn>
const BoundingBox * KdNTree<T, Tn>::ropes() const
{ return m_ropes; }

template <typename T, typename Tn>
BoundingBox * KdNTree<T, Tn>::ropesR(const int & idx)
{ return &m_ropes[idx]; }

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
	
	KdTreeNode * r = root()->node(0);
	if(r->isLeaf() ) return 1;
	
	ctx->setBBox(b);
	int branchIdx = root()->internalOffset(0);
	int nodeIdx = firstVisit(ctx, r);
	int stat;
	bool hasNext = true;
	while (hasNext) {
		stat = visitLeaf(ctx, branchIdx, nodeIdx);
		if(stat > 0 ) {
			hasNext = false;
		}
		else if(stat==0) {
			hasNext = climbRope(ctx, branchIdx, nodeIdx);
		}
	}
	return ctx->m_success;
}

template <typename T, typename Tn>
int KdNTree<T, Tn>::visitLeaf(IntersectionContext * ctx, 
									int & branchIdx,
									int & nodeIdx)
{
	std::cout<<"\n node "<<nodeIdx;
				
	const Tn * branch = nodes()[branchIdx];
	const KdTreeNode * r = branch->node(nodeIdx);
	if(r->isLeaf() ) {
		std::cout<<"\n hit leaf "<<r->getPrimStart();
		if(r->getNumPrims() < 1) {
			return 0;
		}
		std::cout<<" n prim "<<r->getNumPrims();
		ctx->getBBox().intersect(ctx->m_ray, &ctx->m_tmin, &ctx->m_tmax);
		ctx->m_hitP = ctx->m_ray.travel(ctx->m_tmin);
/// leaf ind actually
		ctx->m_componentIdx = r->getPrimStart();
		ctx->m_success = 1;
		return 1;
	}
	
	const int offset = branch->internalOffset(nodeIdx);
	if(r->getOffset() < Tn::TreeletOffsetMask) {
		// std::cout<<"\n inner offset "<<offset;
		nodeIdx += offset + firstVisit(ctx, r);
	}
	else {
		branchIdx += offset;
		std::cout<<"\n branch "<<branchIdx;
		nodeIdx = firstVisit(ctx, r);
	}
	
	return -1;
}

template <typename T, typename Tn>
bool KdNTree<T, Tn>::climbRope(IntersectionContext * ctx, 
									int & branchIdx,
									int & nodeIdx)
{
	const BoundingBox & b = ctx->getBBox();
	float t0, t1;
	b.intersect(ctx->m_ray, &t0, &t1);
	const Vector3F hit1 = ctx->m_ray.travel(t1);
	int side = b.pointOnSide(hit1);
	// std::cout<<"\n rope side "<<side;
	
	const Tn * branch = nodes()[branchIdx];
	const KdTreeNode * r = branch->node(nodeIdx);
/// leaf ind actually 
	int iLeaf = r->getPrimStart();
	std::cout<<"\n side "<<side;
	
	int iRope = leafRopeInd(iLeaf, side);
	
	if(iRope < 0) {
		std::cout<<" no rope";
		return false;
	}
	
	std::cout<<" rope["<<iRope<<"]";
	const BoundingBox & rp = m_ropes[ iRope ];
	BoxNeighbors::DecodeTreeletNodeHash(rp.m_padding1, Tn::BranchingFactor, 
					branchIdx, nodeIdx);
	std::cout<<"\n branch "<<branchIdx;
	ctx->setBBox(rp);
	return true;
}

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

}
//:~

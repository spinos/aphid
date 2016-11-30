/*
 *  ebp.h
 *  
 *
 *  Created by jian zhang on 11/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_EBP_H
#define APH_EBP_H
#include <AdaptiveGrid3.h>
#include <Array.h>
#include <boost/scoped_array.hpp>

namespace aphid {

template<typename Tind, typename Tsrc, typename Tprim>
class PrimInd : public BoundingBox {
	
	Tind * m_ind;
	const Tsrc * m_src;
	
public:
	PrimInd(Tind * ind, const Tsrc * src)
	{
		m_ind = ind;
		m_src = src;
		const Tsrc & rsrc = *src;
		m_ind->begin();
		while(!m_ind->end() ) {
			
			const Tprim * t = rsrc[m_ind->key() ];
			expandBy(t->calculateBBox() );
			
			m_ind->next();
		}
	}
	
	bool intersect(const BoundingBox & box);
	
};

template<typename Tind, typename Tsrc, typename Tprim>
bool PrimInd<Tind, Tsrc, Tprim>::intersect(const BoundingBox & box)
{
	if(!box.intersect(*this) ) return false;
	
	const Tsrc & rsrc = *m_src;
	m_ind->begin();
	while(!m_ind->end() ) {
		
		const Tprim * t = rsrc[m_ind->key() ];
		if(box.intersect(t->calculateBBox() ) ) {
		if(t-> template exactIntersect<BoundingBox >(box) )
			return true;
		}
		
		m_ind->next();
	}
	return false;
}

class EbpNode {

public:
	EbpNode()
	{}
	
	~EbpNode();
	
	Vector3F pos;
	int index;
	
private:

};

EbpNode::~EbpNode()
{}

class EbpCell : public sdb::Array<int, EbpNode >, public sdb::AdaptiveGridCell {

typedef sdb::Array<int, EbpNode > TParent;
	
public:
	EbpCell(Entity * parent = NULL);
	virtual ~EbpCell()
	{}
	
	virtual void clear();
	
private:

};

EbpCell::EbpCell(Entity * parent) :
sdb::Array<int, EbpNode >(parent)
{}

void EbpCell::clear()
{ 
	TParent::clear();
}

class EbpGrid : public sdb::AdaptiveGrid3<EbpCell, EbpNode, 10 > {

typedef sdb::AdaptiveGrid3<EbpCell, EbpNode, 10 > TParent;

	boost::scoped_array<Vector3F > m_pos;
	float m_repelDistance;
	
public:
	EbpGrid()
	{}
	virtual ~EbpGrid()
	{}
	
/// reset level0 cell size and bound
	void fillBox(const BoundingBox & b,
				const float & h);
		
	template<typename Tf>
	void subdivideToLevel(Tf & fintersect,
						int minLevel, int maxLevel)
	{
		BoundingBox cb;
		int level = minLevel;
		while(level < maxLevel) {
			std::vector<sdb::Coord4> dirty;
			begin();
			while(!end() ) {
				if(key().w == level) {
					getCellBBox(cb, key() );
					
					if(fintersect.intersect(cb) )
						dirty.push_back(key() );
				}
				next();
			}
			
			// std::cout<<"\n level"<<level<<" divd "<<dirty.size();
			
			std::vector<sdb::Coord4>::const_iterator it = dirty.begin();
			for(;it!=dirty.end();++it) {
				subdivideCell(fintersect, *it);
			}
			level++;
		}
		m_repelDistance = .67f / levelCellSize(level);
		storeCellNeighbors();
	}
	
	void insertNodeAtLevel(int level);
	void cachePositions();
	const Vector3F * positions() const;
	
	const int & numParticles() const;
	
	void update();
	virtual void clear(); 
			
private:
	template<typename Tf>
	void subdivideCell(Tf & fintersect,
						const sdb::Coord4 & cellCoord)
	{
		EbpCell * cell = findCell(cellCoord);
		if(!cell) {
			std::cout<<"\n [ERROR] EbpGrid cannot find cell to subdivide "<<cellCoord;
			return;
		}
		
		if(cell->hasChild() ) 
			return;
			
		BoundingBox cb;
		int i;	
	/// add 8 children
		for(i=0; i< 8; ++i) { 
			getCellChildBox(cb, i, cellCoord );
			
			if(fintersect.intersect(cb) )
				subdivide(cell, cellCoord, i);
		}
	}
	
	void extractPos(Vector3F * dst);
	void repelForce(Vector3F & frepel,
					EbpCell * cell,
					const EbpNode * node);
	void repelForceInCell(Vector3F & frepel,
					EbpCell * cell,
					const EbpNode * node);
	
};

void EbpGrid::fillBox(const BoundingBox & b,
				const float & h)
{
	clear();
	setLevel0CellSize(h);
	
	const int s = level0CoordStride();
	const sdb::Coord4 lc = cellCoordAtLevel(b.getMin(), 0);
	const sdb::Coord4 hc = cellCoordAtLevel(b.getMax(), 0);
	const int dimx = (hc.x - lc.x) / s + 1;
	const int dimy = (hc.y - lc.y) / s + 1;
	const int dimz = (hc.z - lc.z) / s + 1;
	const float fh = finestCellSize();
	
	const Vector3F ori(fh * (lc.x + s/2),
						fh * (lc.y + s/2),
						fh * (lc.z + s/2));
						
	int i, j, k;
	sdb::Coord4 sc;
	sc.w = 0;
	for(k=0; k<dimz;++k) {
		sc.z = lc.z + s * k;
			for(j=0; j<dimy;++j) {
			sc.y = lc.y + s * j;
			for(i=0; i<dimx;++i) {
				sc.x = lc.x + s * i;
				EbpCell * cell = findCell(sc);
				if(!cell) { 
					addCell(sc);
				}
				
			}
		}
	}
	
	calculateBBox();
}

void EbpGrid::insertNodeAtLevel(int level)
{
	const float hgz = levelCellSize(level) * .49f;
	begin();
	while(!end() ) {
		if(key().w == level) {
			EbpNode * par = new EbpNode;
			
			Vector3F r(RandomFn11(), RandomFn11(), RandomFn11() );
			r.normalize();
			
			par->pos = cellCenter(key() ) + r * (hgz * RandomF01()) ;
			par->index = -1;
			value()->insert(0, par);
			
		}
		next();
	}
}

void EbpGrid::extractPos(Vector3F * dst)
{
	begin();
	while(!end() ) {
		
		EbpCell * cell = value();
		cell->begin();
		while(!cell->end() ) {
			
			const EbpNode * node = cell->value();
			dst[node->index] = node->pos;
			cell->next();
		}
		
		next();
	}
}

void EbpGrid::cachePositions()
{
	const int n = countNodes();
	m_pos.reset(new Vector3F[n]);
	extractPos(m_pos.get());
}

const Vector3F * EbpGrid::positions() const
{ return m_pos.get(); }

const int & EbpGrid::numParticles() const
{ return numNodes(); }

void EbpGrid::update()
{
	Vector3F frepel;
	begin();
	while(!end() ) {
		
		EbpCell * cell = value();
		cell->begin();
		while(!cell->end() ) {
			
			EbpNode * node = cell->value();
			
			frepel.set(0.f,0.f,0.f);
			repelForce(frepel, cell, node);
			repelForceInCell(frepel, cell, node);
			
			node->pos += frepel;
			
			cell->next();
		}
		
		next();
	}
	
	extractPos(m_pos.get());
	
	std::cout.flush();
}

void EbpGrid::repelForce(Vector3F & frepel,
						EbpCell * cell,
						const EbpNode * node)
{
	int i=0;
	for(;i<cell->numNeighbors();++i) {
		repelForceInCell(frepel, 
					static_cast<EbpCell *>(cell->neighbor(i) ), node);
	}
}

void EbpGrid::repelForceInCell(Vector3F & frepel,
					EbpCell * cell,
					const EbpNode * node)
{
	if(!cell) return;
	
	Vector3F vd;
	float l;
	cell->begin();
	while(!cell->end() ) {
		
		const EbpNode * nei = cell->value();
		
		if(nei->index != node->index) {
			vd = node->pos - m_pos.get()[nei->index];
			l = vd.length();
			vd /= l;
			l *= m_repelDistance;
			frepel += vd * std::exp(-8.f*l*l);
		}
		
		cell->next();
	}
}

void EbpGrid::clear()
{
	TParent::clear(); 
}

}
#endif
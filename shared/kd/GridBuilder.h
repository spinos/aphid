/*
 *  GridBuilder.h
 *  julia
 *
 *  Created by jian zhang on 4/8/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <sdb/CartesianGrid.h>
#include <IntersectionContext.h>
#include <Morton3D.h>
#include <kd/KdEngine.h>

namespace aphid {

template<typename T, typename Tn>
class GridBuilder {
	
/// level 3 - 8
	CartesianGrid m_levelGrid[6];
	int m_finalLevel;
	
public:
	GridBuilder();
	virtual ~GridBuilder();
	
	void build(KdNTree<T, Tn > * tree, 
					const BoundingBox & b,
					const int & maxLevel,
					const int & minNPrims);
	
	const int & finalLevel() const;
	int numFinalLevelCells();
	sdb::CellHash * finalLevelCells();
	
protected:
	bool refine(KdNTree<T, Tn > * tree,
				const BoundingBox & b,
				const int & level,
				const int & minNPrims);
	
private:
	
};

template<typename T, typename Tn>
GridBuilder<T, Tn>::GridBuilder()
{}

template<typename T, typename Tn>
GridBuilder<T, Tn>::~GridBuilder()
{}

template<typename T, typename Tn>	
void GridBuilder<T, Tn>::build(KdNTree<T, Tn > * tree, 
					const BoundingBox & b,
					const int & maxLevel,
					const int & minNPrims)
{
	int level = 3;
	
	CartesianGrid & g3 = m_levelGrid[level - 3];
	g3.setBounding(b);
	const int dim = 1<<level;
    int i, j, k;

    const float h = g3.cellSizeAtLevel(level);
    const float hh = h * .49995f;
	
	const Vector3F ori = g3.origin() + Vector3F(hh, hh, hh);
    Vector3F sample;
    BoxIntersectContext box;
	KdEngine eng;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* (float)i, h* (float)j, h* (float)k);
                box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
                box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
				box.reset(minNPrims, true);
				eng.intersectBox<T, Tn>(tree, &box);
				if(box.numIntersect() > minNPrims-1 )
					g3.addCell(sample, level, 1);
            }
        }
    }
	
	m_finalLevel = level;
	
	if(g3.numCells() < 1) return;
	
	while(level < maxLevel) {
		std::cout<<"\r level"<<level<<" n cell "<<m_levelGrid[level - 3].numCells();
		std::cout.flush();
		
		refine(tree, b, level, minNPrims);
		level++;
	}
	m_finalLevel = level;
}

template<typename T, typename Tn>	
bool GridBuilder<T, Tn>::refine(KdNTree<T, Tn > * tree,
				const BoundingBox & b,
				const int & level,
				const int & minNPrims)
{
	sdb::CellHash * c = m_levelGrid[level - 3].cells();
	CartesianGrid & gd = m_levelGrid[level + 1 - 3];
	gd.setBounding(b);
	BoxIntersectContext box;
	KdEngine eng;
	c->begin();
    while(!c->end()) {
        
		int level1 = c->value()->level + 1;
		float hh = gd.cellSizeAtLevel(level1) * .49995f;
		Vector3F sample = gd.cellCenter(c->key() );
		
		for(int u = 0; u < 8; u++) {
			Vector3F subs = sample + Vector3F(hh * gd.Cell8ChildOffset[u][0], 
												hh * gd.Cell8ChildOffset[u][1], 
												hh * gd.Cell8ChildOffset[u][2]);
			box.setMin(subs.x - hh, subs.y - hh, subs.z - hh);
			box.setMax(subs.x + hh, subs.y + hh, subs.z + hh);
			box.reset(minNPrims, true);
			eng.intersectBox<T, Tn>(tree, &box);
			if(box.numIntersect() > minNPrims-1) 
				gd.addCell(subs, level1, 1);
		}
		
        c->next();
	}
	
	return true;
}

template<typename T, typename Tn>
const int & GridBuilder<T, Tn>::finalLevel() const
{ return m_finalLevel; }

template<typename T, typename Tn>
int GridBuilder<T, Tn>::numFinalLevelCells()
{ return m_levelGrid[m_finalLevel - 3].numCells(); }

template<typename T, typename Tn>	
sdb::CellHash * GridBuilder<T, Tn>::finalLevelCells()
{ return m_levelGrid[m_finalLevel - 3].cells(); }

}
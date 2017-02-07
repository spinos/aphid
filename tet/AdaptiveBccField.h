/*
 *  AdaptiveBccField.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "AdaptiveBccMesher.h"
#include <graph/ADistanceField.h>
#include <geom/ConvexShape.h>

namespace aphid {
namespace ttg {

class AdaptiveBccField : public AdaptiveBccMesher, public aphid::ADistanceField {

	aphid::sdb::Sequence<int > m_frontNodes;
	aphid::Vector3F m_insideOutsidePref;
	float m_errorThreshold;
	
public:
	AdaptiveBccField();
	virtual ~AdaptiveBccField();
	
/// subdive grid until either max level and max error condition is met
/// levelLimit < adaptive grid maxLevel
/// threshold is min error distance triggers next level
	template<typename Tf>
	void adaptiveBuild(Tf * distanceFunc,
				const int & levelLimit, 
				const float & threshold)
	{
		m_errorThreshold = threshold;
		
		buildGrid();
		buildMesh();
		buildGraph();
		
		calculateDistance<Tf>(distanceFunc, 0.f);
		obtainGridNodeVal<AdaptiveBccGrid3, BccNode3 >(nodes(), grid() );
		updateMinMaxError();
		
		verbose();
		
		float curErr = maxError();
		
		int level = 0;
		while(level < levelLimit && curErr > threshold) {
			
			std::cout<<"\n subdiv level"<<level<<std::endl;
			subdivideGridByError(threshold, level);
			
			std::cout<<"\n build grid level"<<level<<std::endl;
			buildGrid();
			buildMesh();
			buildGraph();
			
			calculateDistance<Tf>(distanceFunc, 0.f);
			
			if(level+1 < levelLimit) {
				obtainGridNodeVal<AdaptiveBccGrid3, BccNode3 >(nodes(), grid() );
				updateMinMaxError();
			}
			verbose();
		
			level++;
			curErr = maxError();
			
		}
		// printErrEdges(threshold);
		std::cout<<"\n build to level "<<level<<"\n";
		std::cout.flush();
	}

/// grid is very coarse relative to input mesh, error will be large
/// first subdivide to level ignoring error	
/// subdivide front to find negative distance
	template<typename Tf>
	void frontAdaptiveBuild(Tf * distanceFunc, 
							int discreteLevel,
							int maxLevel,
							const float & thre)
	{
		int curLevel = discreteLevel, nbound;
		discretize<Tf>(distanceFunc, curLevel);
		
		buildGrid();
		buildMesh();
		buildGraph();
		
		float thickness = grid()->levelCellSize(curLevel) * thre;
		distanceFunc->setShellThickness(thickness );
		calculateDistance2<Tf>(distanceFunc);
		verbose();
		
		while(curLevel < maxLevel) {
			obtainGridNodeVal<AdaptiveBccGrid3, BccNode3 >(nodes(), grid() );
		
			std::cout<<"\n subdiv level"<<curLevel<<std::endl;
		
			subdivideByFront(curLevel);
			nbound = m_frontNodes.size();
			std::cout<<"\n n boundary nodes "<<nbound;
			
			buildGrid();
			buildMesh();
			buildGraph();
			
			thickness = grid()->levelCellSize(curLevel+1) * thre;
			distanceFunc->setShellThickness(thickness);
			moveFront(thickness);
			calculateDistance2<Tf>(distanceFunc);
		
			std::cout<<"\n subdiv to level"<<curLevel;
			verbose();
			
			curLevel++;
		}
		
	}

/// block edges by split tetrahedron
/// detect front cells by distance sign changes and geom intersect
/// only front cells are subdivided
/// skip interior (all negative) cells
	template<typename Tf>
	void subdivideFront(Tf * distanceFunc, 
							int curLevel)
	{
		std::cout<<"\n subdiv level"<<curLevel<<std::endl;
		AdaptiveBccGrid3 * g = grid();
		
/// key to all front cells
		std::vector<aphid::sdb::Coord4 > frontCells;
		
		aphid::sdb::Coord4 k;
		aphid::BoundingBox dirtyBx;
		
		g->begin();
		while(!g->end() ) {
		
			k = g->key();
			if(k.w == curLevel) {
				if(g->value()->isFront(k, g) ) {
					frontCells.push_back(k);
				}
				else if (!g->value()->isInterior(k, g) ) {
				
					g->getCellBBox(dirtyBx, k);
					
/// not interior but intersect 
					if(distanceFunc-> template broadphase <aphid::BoundingBox>(&dirtyBx )) 
						frontCells.push_back(k);
				}
			}
			
			g->next();
		}
		
		std::vector<aphid::sdb::Coord4 > divided;
		
		const int level1 = curLevel+1;
		std::vector<aphid::sdb::Coord4 >::const_iterator it = frontCells.begin();
		for(;it!=frontCells.end();++it) {
			g->subdivideCell(*it, &divided);
			
		}
		
		enforceBoundary(divided);
		divided.clear();
		
		std::cout<<"\n n front cell "<<frontCells.size();
	
		frontCells.clear();
	}
	
	template<typename Tf>
	void marchFrontBuild(Tf * distanceFunc, 
							int maxLevel)
	{
		int curLevel = 3, nbound;
		discretize<Tf>(distanceFunc, curLevel);
		
		buildGrid();
		buildMesh();
		buildGraph();
		
		calculateDistance2<Tf>(distanceFunc);
		obtainGridNodeVal<AdaptiveBccGrid3, BccNode3 >(nodes(), grid() );
		
		verbose();
		
		while(curLevel < maxLevel) {
			
			subdivideFront(distanceFunc, curLevel);
			
			buildGrid();
			buildMesh();
			buildGraph();
			
			calculateDistance2<Tf>(distanceFunc);
			obtainGridNodeVal<AdaptiveBccGrid3, BccNode3 >(nodes(), grid() );
			
			verbose();
			
			std::cout<<"\n subdiv to level"<<++curLevel;
		}
	}
	
	void buildGrid();
	
/// push tetra node and edge to graph
	void buildGraph();
	
	template<typename Tf>
	void calculateDistance(Tf * func, const float & shellThickness)
	{
		clearDirtyEdges();
		markUnknownNodes();
		
		typename aphid::cvx::Tetrahedron;
		aphid::cvx::Tetrahedron tetshp;

		const int nt = numTetrahedrons();
		int i = 0;
		for(;i<nt;++i) {
			
			getTetraShape(tetshp, i);

/// intersect any tetra			
			if(func-> template broadphase <aphid::cvx::Tetrahedron>(&tetshp ) ) {
				markTetraOnFront(i);
			}
		}
		
		messureFrontNodes(func, shellThickness);
		fastMarchingMethod();
		int ifar = findFarInd();
		markInsideOutside2(func, ifar, true);
		estimateFrontEdgeError(func);
		
	}
	
/// split tetrahedron to four hexahedron
/// if hexahedron intersect, block three edges
	template<typename Tf>
	void findTetraEdgeCross(Tf * func, 
							const int & itet,
							const aphid::cvx::Hexahedron * hexashp) 
	{
		if(func-> narrowphase (hexashp[0]) ) {
			setTetraVertexEdgeCross(itet, 0, .5f);
		}

		if(func-> narrowphase (hexashp[1]) ) {
			setTetraVertexEdgeCross(itet, 1, .5f);
		}
		
		if(func-> narrowphase (hexashp[2]) ) {
			setTetraVertexEdgeCross(itet, 2, .5f);
		}
		
		if(func-> narrowphase (hexashp[3]) ) {
			setTetraVertexEdgeCross(itet, 3, .5f);
		}
	}
	
	template<typename Tf>
	void calculateDistance2(Tf * func)
	{
		std::cout<<"\n AdaptiveBccField::calculateDistance2 begin"<<std::endl;
		markUnknownNodes();
		
		typename aphid::cvx::Tetrahedron;
		aphid::cvx::Tetrahedron tetshp;
		aphid::cvx::Hexahedron hexashp[4];
		
		const int nt = numTetrahedrons();
		int i = 0;
		for(;i<nt;++i) {
			
			getTetraShape(tetshp, i);

/// intersect any tetra			
			if(func-> template broadphase <aphid::cvx::Tetrahedron>(&tetshp ) ) {
				markTetraNodeOnFront(i);
				tetshp.split(hexashp);

				findTetraEdgeCross(func, i, hexashp);
			}
		}
		
		messureFrontNodes2(func);
		int ifar = findFarInd();
		markInsideOutside(ifar);
		fastMarchingMethod();
		std::cout<<"\n AdaptiveBccField::calculateDistance2 end"<<std::endl;
		
	}
	
	void verbose();
	
	void getTetraShape(aphid::cvx::Tetrahedron & b, const int & i) const;
	void setInsideOutsidePref(const aphid::Vector3F & q);
	const float & errorThreshold() const;
								
protected:
	void markTetraNodeOnFront(const int & i);
/// four nodes on front
/// six edges dirty
	void markTetraOnFront(const int & i);
	
private:
	void pushIndices(const std::vector<int> & a,
					std::vector<int> & b);
	void subdivideGridByError(const float & threshold,
						const int & level);
	void subdivideByFront(int level);
/// background node ind in cell closest to pref 
	int findFarInd();
	bool isTetraOnFront(int iv0, int iv1, int iv2, int iv3) const;
	bool isTetraOnFrontBoundary(int iv0, int iv1, int iv2, int iv3) const;
	bool isTetraAllPositive(int iv0, int iv1, int iv2, int iv3) const;
	void moveFront(const float & x);
	void setTetraVertexEdgeCross(const int & itet,
								const int & ivertex,
								const float & val);
	
};

}
}
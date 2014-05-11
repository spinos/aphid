/*
 *  Sculptor.h
 *  btree
 *
 *  Created by jian zhang on 5/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <C3Tree.h>
#include <RayMarch.h>
namespace sdb {
class Sculptor {
public:
	class ActiveGroup {
	public:
		ActiveGroup();
		
		void reset();
		
		int numSelected();
		
		float depthRange();
		
		void updateDepthRange(const float & d);
		
		void finish();
		
		void average(const List<VertexP> * d);
		
		Ordered<int, VertexP> * vertices;
		float depthMin, depthMax, gridSize, threshold;
		Vector3F meanPosition, meanNormal;
		int numActivePoints, numActiveBlocks;
	};
	
	Sculptor();
	virtual ~Sculptor();
	
	void beginAddVertices(const float & gridSize);
	void addVertex(const VertexP & v);
	void endAddVertices();
	
	void setSelectRadius(const float & x);
	
	void selectPoints(const Ray * incident);
	void deselectPoints();
	
	C3Tree * allPoints() const;
	ActiveGroup * activePoints() const;
	
	void pullPoints();
	
private:
	bool intersect(List<VertexP> * ps, const Ray & ray);
private:
	RayMarch m_march;
	ActiveGroup * m_active;
	C3Tree * m_tree;
};
} // end namespace sdb
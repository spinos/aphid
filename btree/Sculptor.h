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
	struct ActiveGroup {
		ActiveGroup() { vertices = new Ordered<int, VertexP>; reset(); }
		
		void reset() {
			depthMin = 10e8;
			depthMax = -10e8;
			vertices->clear();
		}
		
		int numSelected() { return vertices->numElements(); }
		
		float depthRange() {
			return depthMax - depthMin;
		}
		
		void updateDepthRange(const float & d) {
			if(d > depthMax) depthMax = d;
			if(d < depthMin) depthMin = d;
		}
		
		Ordered<int, VertexP> * vertices;
		float depthMin, depthMax, gridSize, threshold;
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
	
private:
	bool intersect(List<VertexP> * ps, const Ray & ray);
private:
	RayMarch m_march;
	ActiveGroup * m_active;
	C3Tree * m_tree;
};
} // end namespace sdb
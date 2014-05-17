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
#include <Dropoff.h>
#include <MeshTopology.h>
namespace sdb {
class Sculptor {
public:
	class ActiveGroup {
	public:
		ActiveGroup();
		~ActiveGroup();
		void reset();
		int numSelected();
		float depthRange();
		void updateDepthRange(const float & d);
		void finish();
		void average(List<VertexP> * d);
		const float weight(const int & i) const;
		void setDropoffFunction(Dropoff::DistanceFunction x);
		const int numActivePoints() const;
		const int numActiveBlocks() const;
		const float meanDepth() const;
		
		Ordered<int, VertexP> * vertices;
		float depthMin, depthMax, gridSize, threshold;
		Vector3F meanPosition, meanNormal;
		Ray incidentRay;
	private:
	    void calculateWeight();
	    void calculateWeight(List<VertexP> * d);
	    std::deque<float> m_weights;
	    Dropoff::DistanceFunction m_dropoffType;
	    Dropoff *m_drop;
		int m_numActivePoints, m_numActiveBlocks;
	};
	
	Sculptor();
	virtual ~Sculptor();
	
	void beginAddVertices(const float & gridSize);
	void addVertex(const VertexP & v);
	void endAddVertices();
	
	void setSelectRadius(const float & x);
	const float selectRadius() const;
	
	void setStrength(const float & x);
	
	void setMeshTopology(MeshTopology * topo);
	
	void selectPoints(const Ray * incident);
	void deselectPoints();
	
	C3Tree * allPoints() const;
	ActiveGroup * activePoints() const;
	
	void pullPoints();
	void pushPoints();
	void pinchPoints();
	void smudgePoints(const Vector3F & x);
	void smoothPoints();
	
private:
	bool intersect(List<VertexP> * ps, const Ray & ray);
	void movePointsAlong(const Vector3F & d, const float & fac);
	void movePointsToward(const Vector3F & d, const float & fac);
private:
	RayMarch m_march;
	ActiveGroup * m_active;
	C3Tree * m_tree;
	MeshTopology * m_topo;
	float m_strength;
};
} // end namespace sdb
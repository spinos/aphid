/*
 *  ComponentConversion.h
 *  knitfabric
 *
 *  Created by jian zhang on 6/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllGeometry.h>
#include <vector>
class MeshTopology;

class ComponentConversion {
public:
	ComponentConversion();
	
	void setTopology(MeshTopology * topo);
	
	void edgeRing(const unsigned & src, std::vector<unsigned> & edgeIds) const;
	void facetToPolygon(const std::vector<unsigned> & src, std::vector<unsigned> & polyIds) const;
	void edgeToVertex(const std::vector<unsigned> & src, std::vector<unsigned> & dst) const;
	void vertexToEdge(const std::vector<unsigned> & src, std::vector<unsigned> & dst) const;
	void vertexToEdge(const std::vector<unsigned> & src, std::vector<Edge *> & dst) const;
	void vertexToPolygon(const std::vector<unsigned> & src, std::vector<unsigned> & polyIds, std::vector<unsigned> & vppIds) const;
	void edgeToPolygon(const std::vector<unsigned> & src, std::vector<unsigned> & polyIds, std::vector<unsigned> & vppIds) const;
	Vector3F vertexPosition(unsigned idx) const;
private:
	char appendUnique(unsigned idx, std::vector<unsigned> & dst) const;
	MeshTopology * m_topology;
};
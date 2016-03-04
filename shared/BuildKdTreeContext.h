/*
 *  BuildKdTreeContext.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <SplitEvent.h>
#include <BuildKdTreeStream.h>
#include <GridClustering.h>

namespace aphid {

class BuildKdTreeContext {

	GridClustering * m_grid;
	
public:
	BuildKdTreeContext();
	BuildKdTreeContext(BuildKdTreeStream &data, const BoundingBox & b);
	~BuildKdTreeContext();
	
	void createGrid(const float & x);
	
	const unsigned & getNumPrimitives() const;

	const sdb::VectorArray<unsigned> & indices() const;
	const sdb::VectorArray<BoundingBox> & primitiveBoxes() const;
	
	void verbose() const;

	void setBBox(const BoundingBox &bbox);
	const BoundingBox & getBBox() const;
	float visitCost() const;
	bool isCompressed();
	sdb::WorldGrid<GroupCell, unsigned > * grid();
	void addCell(const sdb::Coord3 & x, GroupCell * c);
	void countPrimsInGrid();
	int numCells();
	bool decompress(bool forced = false);
	void addIndex(const unsigned & x);
	
	static BuildKdTreeContext * GlobalContext;
	
private:
	
private:
	BoundingBox m_bbox;
	sdb::VectorArray<unsigned> m_indices;
	sdb::VectorArray<BoundingBox> m_primitiveBoxes;
	unsigned m_numPrimitive;
};

}
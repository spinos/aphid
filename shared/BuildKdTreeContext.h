/*
 *  BuildKdTreeContext.h
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BoundingBox.h>
#include <IndexList.h>
#include <SplitEvent.h>
#include <BuildKdTreeStream.h>
#include <WorldGrid.h>
#include <VectorArray.h>

namespace aphid {

class GroupCell : public sdb::Sequence<unsigned>
{
public:
	GroupCell(sdb::Entity * parent = NULL) : sdb::Sequence<unsigned>(parent) {}
	BoundingBox m_box;
};

class BuildKdTreeContext {
	sdb::WorldGrid<GroupCell, unsigned > * m_grid;
	
public:
	BuildKdTreeContext();
	BuildKdTreeContext(BuildKdTreeStream &data, const BoundingBox & b);
	~BuildKdTreeContext();
	
	void createIndirection(const unsigned &count);
	void createGrid(const float & x);
	
	const unsigned & getNumPrimitives() const;

	unsigned *indices();
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
	
private:
	void addIndices(GroupCell * c, unsigned & ind);
	
private:
	BoundingBox m_bbox;
	IndexList m_indices;
	sdb::VectorArray<BoundingBox> m_primitiveBoxes;
	unsigned m_numPrimitive;
};

}
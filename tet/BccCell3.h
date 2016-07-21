/*
 *  BccCell3.h
 *  foo
 *
 *  Created by jian zhang on 7/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Array.h>
#include <AdaptiveGrid3.h>
#include "BccCell.h"

namespace ttg {

class BccCell3 : public aphid::sdb::Array<int, BccNode > {

	bool m_hasChild;
	
public:
	BccCell3(Entity * parent = NULL);
	virtual ~BccCell3();
	
	void setHasChild();
	void insertRed(const aphid::Vector3F & pref);
	void insertBlue(const aphid::sdb::Coord4 & cellCoord,
					aphid::sdb::AdaptiveGrid3<BccCell3, BccNode > * grid);
	BccNode * findBlue(const aphid::Vector3F & pref);
	void insertFaceOnBoundary(const aphid::sdb::Coord4 & cellCoord,
					aphid::sdb::AdaptiveGrid3<BccCell3, BccNode > * grid);
	
	const bool & hasChild() const;
	
private:
	void insertBlue0(const aphid::Vector3F & center,
					const aphid::sdb::Coord4 & cellCoord,
					aphid::sdb::AdaptiveGrid3<BccCell3, BccNode > * grid);
	void insertBlue1(const aphid::Vector3F & center,
					const aphid::sdb::Coord4 & cellCoord,
					aphid::sdb::AdaptiveGrid3<BccCell3, BccNode > * grid);
				
};

}
/*
 *  NTreeIO.h
 *  
 *
 *  Created by jian zhang on 3/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HDocument.h>
#include <ConvexShape.h>
#include <HNTree.h>
#include <VectorArray.h>

namespace aphid {

class NTreeIO {

public:
	NTreeIO();
	
	bool begin(const std::string & filename, 
				HDocument::OpenMode om = HDocument::oReadOnly);
	void end();
	
	bool findGrid(std::string & name,
				const std::string & grpName="/");
	
	bool findTree(std::string & name,
				const std::string & grpName="/");
	
	cvx::ShapeType gridValueType(const std::string & name);
	
	template<typename T>
	void loadGridCoord(sdb::VectorArray<cvx::Cube> * dst, const std::string & name)
	{
		T grd(name);
		grd.load();
		const float h = grd.gridSize();
/// shrink a little
		const float e = h * .5f;
		cvx::Cube c;
		grd.begin();
		while(!grd.end() ) {
			c.set(grd.coordToCellCenter(grd.key() ), e);
			dst->insert(c);
			grd.next();
		}
		grd.close();
	}
	
protected:

private:

};

}
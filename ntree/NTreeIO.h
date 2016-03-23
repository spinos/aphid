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
#include <HNTree.h>
#include <VectorArray.h>
#include <HElemAsset.h>

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
		loadGridCoord<T>(dst, &grd);
		grd.close();
	}
	
	template<typename T>
	void loadGridCoord(sdb::VectorArray<cvx::Cube> * dst, T * grd)
	{
		const float h = grd->gridSize();
		const float e = h * .5f;
		cvx::Cube c;
		grd->begin();
		while(!grd->end() ) {
			c.set(grd->coordToCellCenter(grd->key() ), e);
			dst->insert(c);
			grd->next();
		}
	}
	
	template<typename T>
	bool findElemAsset(std::string & name,
				const std::string & grpName="/")
	{
		std::vector<std::string > assetNames;
		HBase r(grpName);
		r.lsTypedChildWithIntAttrVal<HElemBase>(assetNames,
											".elemtyp", T::ShapeTypeId );
		r.close();
		
		if(assetNames.size() <1) {
			std::cout<<"\n found no elem";
			return false;
		}
		name = assetNames[0];
		return false;
	}
	
protected:

private:

};

}
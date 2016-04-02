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
#include <HAssetGrid.h>

namespace aphid {

class NTreeIO {

	HDocument m_doc;
	
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
/// no path
		int islash = name.rfind('/', name.size() );
		if(islash >=0)
			name.erase(0, ++islash);
		return true;
	}
	
	template<typename Ta, typename Tv>
	bool extractAsset(const std::string & name, 
							sdb::VectorArray<Tv> * dst,
							BoundingBox & box);
	
	template<typename T>
	bool hasNamedAsset(const std::string & grpName,
						const std::string & name);
	
protected:

private:

};

template<typename Ta, typename Tv>
bool NTreeIO::extractAsset(const std::string & name, 
							sdb::VectorArray<Tv> * dst,
							BoundingBox & box)
{
	Ta ass(name);
	ass.load();
	box = ass.getBBox();
	std::cout<<"\n  bbox "<<box;
	if(ass.numElems() > 0) {
		ass.extract(dst);
		std::cout<<"\n extract n "<<Tv::GetTypeStr()<<" "<<dst->size();
	}
	ass.close();
	return dst->size() > 0;
}

template<typename T>
bool NTreeIO::hasNamedAsset(const std::string & grpName,
						const std::string & name)
{
	HBase g(grpName);
	bool stat = g.hasNamedChild(name.c_str() );
	if(stat) {
		HBase a(g.childPath(name) );
		stat = a.hasNamedAttrIntVal(".elemtyp", T::ShapeTypeId)
				&& a.hasNamedAttrIntVal(".act", 1);
		a.close();
	}
	g.close();
	return stat;
}

}
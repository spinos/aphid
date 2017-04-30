/*
 *  LoadElemAsset.h
 *  
 *	T as element type
 *
 *  Created by jian zhang on 2/8/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_H5_LOAD_ELEM_ASSET_H
#define APH_H5_LOAD_ELEM_ASSET_H

#include <h5/NTreeIO.h>
#include <sdb/VectorArray.h>
#include <geom/ConvexShape.h>

namespace aphid {

template<typename T>
class LoadElemAsset {

public:
	LoadElemAsset();
	
	bool loadSource(sdb::VectorArray<T> * source,
						BoundingBox & bbox,
								const std::string & fileName);

protected:
	void loadTriangles(sdb::VectorArray<T> * source,
						BoundingBox & bbox,
								const std::string & name);
};

template<typename T>
LoadElemAsset<T>::LoadElemAsset()
{}

template<typename T>
bool LoadElemAsset<T>::loadSource(sdb::VectorArray<T> * source,
								BoundingBox & bbox,
								const std::string & fileName)
{
	NTreeIO hio;
	if(!hio.begin(fileName) ) {
		std::cout<<"\n cannot open file "<<fileName;
		
		return false;
	}
	
	std::string elmName;
	bool stat = hio.findElemAsset<T>(elmName);
	if(stat) {
		std::cout<<"\n found "<<T::GetTypeStr()<<" type asset "<<elmName;
		
		if(T::ShapeTypeId == cvx::TTriangle ) {
			loadTriangles(source, bbox, elmName);
		}
	}
	else {
		std::cout<<"\n found no "<<T::GetTypeStr()<<" type asset";
	}
	
	hio.end();
	
	return stat;
	
}

template<typename T>
void LoadElemAsset<T>::loadTriangles(sdb::VectorArray<T> * source,
								BoundingBox & bbox,
								const std::string & name)
{
	HTriangleAsset ass(name);
	ass.load();
	bbox = ass.getBBox();
	std::cout<<"\n world bbox "<<bbox;
	if(ass.numElems() > 0) {
		ass.extract(source);
		std::cout<<"\n n tri "<<source->size();
	}
	ass.close();
}

}

#endif

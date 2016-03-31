/*
 *  JuliaWorld.cpp
 *  julia
 *
 *  Created by jian zhang on 3/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "JuliaWorld.h"
#include <HInnerGrid.h>
#include <HWorldGrid.h>
#include <KdEngine.h>
#include <NTreeIO.h>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
using namespace aphid;

namespace jul {

JuliaWorld::JuliaWorld() {}

JuliaWorld::~JuliaWorld() {}

void JuliaWorld::create(const Parameter & param) 
{
	if(boost::filesystem::exists(param.outFileName() ) ) {
		std::cout<<"\n file already exists "<<param.outFileName()
		<<"\n are you sure to override it? ";
		
		char b[256];
		std::cin >> b;
		
		if(strcmp(b, "y") == 0 || strcmp(b, "yes") == 0) 
		{}
		else {
			std::cout<<"\n exit ";
			return;
		}
	}
	
	std::cout<<"\n initialize world "
		<<"\n cell size "<<param.cellSize();
		
	NTreeIO hio;
	bool stat = hio.begin(param.outFileName(), HDocument::oCreate );
	if(!stat) return;
		
	sdb::HWorldGrid<sdb::HAssetGrid<HTriangleAsset, cvx::Triangle >, cvx::Triangle > wg("/grid");
	wg.setGridSize((float)param.cellSize() );
	
	wg.save();
	wg.close();
	
	hio.end();
}

void JuliaWorld::insert(const Parameter & param)
{
	NTreeIO hio;
	bool stat = hio.begin(param.inFileName(), HDocument::oReadOnly );
	if(!stat) return;
	
	std::cout<<"\n insert asset "<<param.inFileName();

	sdb::VectorArray<cvx::Triangle> source;
	BoundingBox srcBox;
	std::string elmName;
	stat = hio.findElemAsset<cvx::Triangle>(elmName);
	if(stat) {
		std::cout<<"\n found "<<cvx::Triangle::GetTypeStr()<<" type asset "<<elmName;
		
		stat = hio.extractAsset<HTriangleAsset, cvx::Triangle>(elmName, &source, srcBox);
		
	}
	else 
		std::cout<<"\n found no "<<cvx::Triangle::GetTypeStr()<<" type asset";
	
	hio.end();
	
	if(!stat) return;
	
	stat = hio.begin(param.outFileName(), HDocument::oReadAndWrite );
	if(!stat) return;
	
	std::cout<<"\n into world "<<param.outFileName();
	sdb::HWorldGrid<sdb::HAssetGrid<HTriangleAsset, cvx::Triangle >, cvx::Triangle > wg("/grid");
	wg.load();
	
	wg.insert(elmName, srcBox);
	
	const Vector3F borOri = srcBox.getMin();
	const int n = source.size();
	int i=0;
	for(;i<n;++i) {
		wg.insert(source[i], borOri);
	}
	wg.save();
	hio.end();
}

}
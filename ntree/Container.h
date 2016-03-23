/*
 *  Container.h
 *  
 *
 *  Created by jian zhang on 3/23/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <KdEngine.h>
#include <ConvexShape.h>
#include <IntersectionContext.h>
#include <VoxelGrid.h>
#include <NTreeIO.h>

using namespace aphid;

template<typename T>
class Container {

	KdEngine<T> m_engine;
	sdb::VectorArray<T> * m_source;
	KdNTree<T, KdNode4 > * m_tree;
	VoxelGrid<KdNTree<T, KdNode4 >, T > * m_grid;
	
public:
	Container();
	virtual ~Container();
	
	bool readTree(const std::string & filename);
	KdNTree<T, KdNode4 > * tree();
	
protected:

private:
	
};

template<typename T>
Container<T>::Container()
{
	m_source = NULL;
	m_tree = NULL;
	m_grid = NULL;
	
}

template<typename T>
Container<T>::~Container()
{}

template<typename T>
bool Container<T>::readTree(const std::string & filename)
{
	bool stat = false;
	NTreeIO hio;
	if(!hio.begin(filename) ) return false;
	
	std::string elmName;
	//stat = hio.findElemAsset(elmName);
	if(stat) {
		std::cout<<"\n asset "<<elmName;
		m_source = new sdb::VectorArray<T>();
		
	}
	
	hio.end();
	return true;
}

template<typename T>
KdNTree<T, KdNode4 > * Container<T>::tree()
{ return m_tree; }

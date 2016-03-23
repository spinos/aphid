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
	const sdb::VectorArray<T> * source() const;
	
protected:

private:
	void loadTriangles(const std::string & name);
	
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
	stat = hio.findElemAsset<T>(elmName);
	if(stat) {
		std::cout<<"\n found "<<T::GetTypeStr()<<" type asset "<<elmName;
		
		if(T::ShapeTypeId == cvx::TTriangle ) 
			loadTriangles(elmName);
		
	}
	else 
		std::cout<<"\n found no "<<T::GetTypeStr()<<" type asset";
	
	hio.end();
	return true;
}

template<typename T>
KdNTree<T, KdNode4 > * Container<T>::tree()
{ return m_tree; }

template<typename T>
const sdb::VectorArray<T> * Container<T>::source() const
{ return m_source; }

template<typename T>
void Container<T>::loadTriangles(const std::string & name)
{
	HTriangleAsset ass(name);
	ass.load();
	std::cout<<"\n bbox "<<ass.getBBox();
	if(ass.numElems() > 0) {
		m_source = new sdb::VectorArray<T>();
		ass.extract(m_source);
		std::cout<<"\n n tri "<<m_source->size();
	}
	ass.close();
}

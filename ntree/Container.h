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

	BoundingBox m_worldBox;
	sdb::VectorArray<T> * m_source;
	KdNTree<T, KdNode4 > * m_tree;
	VoxelGrid<T, KdNode4 > * m_grid;
	KdNTree<Voxel, KdNode4 > * m_voxelTree;
	
public:
	Container();
	virtual ~Container();
	
	bool readTree(const std::string & filename, int gridLevel = 6);
	KdNTree<T, KdNode4 > * tree();
	KdNTree<Voxel, KdNode4 > * voxelTree();
	VoxelGrid<T, KdNode4 > * grid();
	const sdb::VectorArray<T> * source() const;
	const BoundingBox & worldBox() const;
	
protected:

private:
	void loadTriangles(const std::string & name);
	bool buildTree();
	bool buildGrid(int gridLevel);
	
};

template<typename T>
Container<T>::Container()
{
	m_source = NULL;
	m_tree = NULL;
	m_voxelTree = NULL;
	m_grid = NULL;
	m_worldBox.reset();
}

template<typename T>
Container<T>::~Container()
{}

template<typename T>
bool Container<T>::readTree(const std::string & filename, int gridLevel)
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
	
	buildTree();
	return buildGrid(gridLevel);
}

template<typename T>
bool Container<T>::buildTree()
{
	if(!m_source) return false;
	if(m_source->size() < 1) return false;
	
	float d0 = m_worldBox.distance(0);
	float d1 = m_worldBox.distance(1);
	float d2 = m_worldBox.distance(2);
	BoundingBox rootBox;
	rootBox.setMin(0.f, 0.f, 0.f);
	rootBox.setMax(d0, d1, d2);
	std::cout<<"\n root bbox "<<rootBox;
	
	m_tree = new KdNTree<T, KdNode4 >();
	
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 128;
	
	KdEngine engine;
	engine.buildTree<T, KdNode4, 4>(m_tree, m_source, rootBox, &bf);
	m_tree->setRelativeTransform(rootBox);
	
	return true;
}

template<typename T>
bool Container<T>::buildGrid(int gridLevel)
{
	m_grid = new VoxelGrid<T, KdNode4 >();
	BoundingBox b = m_tree->getBBox();
	
	typename VoxelGrid<T, KdNode4>::BuildProfile vf;
	vf._minNPrimsPerCell = 1;
	vf._maxLevel = gridLevel;
	
	m_grid->create(m_tree, b, &vf);
	
	Vector3F o = m_grid->origin();
	float sp = 1024.f;
	BoundingBox gridBox(o.x, o.y, o.z,
						o.x+sp, o.y+sp, o.z+sp);
	std::cout<<"\n grid box "<<gridBox;
						
	m_voxelTree = new KdNTree<Voxel, KdNode4 >();
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 32;
    bf._doTightBox = true;
    
	KdEngine engine;
	engine.buildTree<Voxel, KdNode4, 4>(m_voxelTree, m_grid->voxelsR(), gridBox, &bf);
	BoundingBox brel;
	m_grid->getBounding(brel);
	m_voxelTree->setRelativeTransform(brel);
    return true;
}

template<typename T>
KdNTree<T, KdNode4 > * Container<T>::tree()
{ return m_tree; }

template<typename T>
KdNTree<Voxel, KdNode4 > * Container<T>::voxelTree()
{ return m_voxelTree; }

template<typename T>
const sdb::VectorArray<T> * Container<T>::source() const
{ return m_source; }

template<typename T>
VoxelGrid<T, KdNode4 > * Container<T>::grid()
{ return m_grid; }

template<typename T>
const BoundingBox & Container<T>::worldBox() const
{ return m_worldBox; }

template<typename T>
void Container<T>::loadTriangles(const std::string & name)
{
	HTriangleAsset ass(name);
	ass.load();
	m_worldBox = ass.getBBox();
	std::cout<<"\n world bbox "<<m_worldBox;
	if(ass.numElems() > 0) {
		m_source = new sdb::VectorArray<T>();
		ass.extract(m_source);
		std::cout<<"\n n tri "<<m_source->size();
	}
	ass.close();
}

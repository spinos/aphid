/*
 *  HAssetGrid.h
 *  julia
 *
 *  Created by jian zhang on 3/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  grid contains a number of element asset
 *  within boundary
 */

#include <HBase.h>
#include <HElemAsset.h>
#include <KdEngine.h>
#include <VoxelGrid.h>

namespace aphid {
namespace sdb {

template <typename T, typename Tv>
class HAssetGrid : public HBase, public Entity {
	
	T * m_activeAsset;
	
public:
	HAssetGrid(const std::string & name, Entity * parent);
	virtual ~HAssetGrid();
	
	bool insert(const Tv * v);
	bool insert(const std::string & name);
	bool flush();
	int numElements();
	bool beginRead();
	void remove(const std::string & name);
	void buildTree(const BoundingBox & worldBox);
	
protected:

private:

};

template <typename T, typename Tv>
HAssetGrid<T, Tv>::HAssetGrid(const std::string & name, Entity * parent) :
HBase(name), Entity(parent),
m_activeAsset(NULL)
{}

template <typename T, typename Tv>
HAssetGrid<T, Tv>::~HAssetGrid()
{}

template <typename T, typename Tv>
bool HAssetGrid<T, Tv>::beginRead()
{
	return true;
}

template <typename T, typename Tv>
int HAssetGrid<T, Tv>::numElements()
{
	int sum = 0;
	int an;
/// count all asset n element
	std::vector<std::string > assetNames;
	lsTypedChildWithIntAttrVal<HElemBase>(assetNames,
											".elemtyp", Tv::ShapeTypeId );
	std::vector<std::string >::const_iterator it = assetNames.begin();
	for(;it != assetNames.end();++it) {
		T ass(*it);
		an = 0;
		ass.readIntAttr(".nelem", &an );
		sum += an;
		ass.close();
	}
	return sum;
}

template <typename T, typename Tv>
bool HAssetGrid<T, Tv>::insert(const std::string & name)
{
/// open named asset
	std::cout<<"\n hasset insert "<<(childPath(name) );
	m_activeAsset = new T(childPath(name) );
	return true;
}

template <typename T, typename Tv>
bool HAssetGrid<T, Tv>::insert(const Tv * v)
{
	m_activeAsset->insert(*v);
	return true;
}

template <typename T, typename Tv>
bool HAssetGrid<T, Tv>::flush()
{
	if(!m_activeAsset) return false;
	m_activeAsset->save();
	m_activeAsset->close();
	delete m_activeAsset;
	return true;
}

template <typename T, typename Tv>
void HAssetGrid<T, Tv>::remove(const std::string & name)
{
	if(hasTypedChildWithIntAttrVal<T>(name, ".elemtyp", Tv::ShapeTypeId ) ) {
		std::cout<<"\n hasset remove "<<(childPath(name) );
		T c(childPath(name) );
		c.clear();
		c.close();
	}
}

template <typename T, typename Tv>
void HAssetGrid<T, Tv>::buildTree(const BoundingBox & worldBox)
{
	sdb::VectorArray<Tv> src;
	std::vector<std::string > assetNames;
	lsTypedChildWithIntAttrVal<HElemBase>(assetNames,
											".elemtyp", Tv::ShapeTypeId );
	std::vector<std::string >::const_iterator it = assetNames.begin();
	for(;it != assetNames.end();++it) {
		T ass(*it);
		if(ass.load() ) {
			ass.extract(&src);
		}
		ass.close();
	}
	
	if(src.size() < 1) return;
	
	std::cout<<"\n hassetgrid extract "<<src.size()<<" "<<Tv::GetTypeStr()
			<<" in world bbox "<<worldBox;
	BoundingBox rootBox(0.f, 0.f, 0.f,
						worldBox.distance(0), 
						worldBox.distance(1), 
						worldBox.distance(2));
	
	KdNTree<Tv, KdNode4 > ptree;
	
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 64;
	
	KdEngine engine;
	engine.buildTree<Tv, KdNode4, 4>(&ptree, &src, rootBox, &bf);
	
	VoxelGrid<KdNTree<Tv, KdNode4 >, Tv > vgd;
	vgd.create(&ptree, rootBox, 8);
	
	if(vgd.numVoxels() < 1) return;
	
}

}
}
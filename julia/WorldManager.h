#ifndef APHID_JUL_WORLD_MANAGER_H
#define APHID_JUL_WORLD_MANAGER_H

/*
 *  WorldManager.h
 *  julia
 *
 *  Created by jian zhang on 4/7/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <Manager.h>
#include <CudaNTree.h>

namespace aphid {

namespace jul {

template<typename WorldT, typename InnerT, typename TreeT>
class WorldManager : public Manager<WorldT, InnerT, TreeT> {

typedef CudaNTree<cvx::Box, KdNode4> WorldTreeT;
	boost::scoped_ptr<WorldTreeT> m_worldTree;
	
public:
	WorldManager();
	virtual ~WorldManager();
	
	virtual bool openWorld(const std::string & filename);
	
	CudaNTree<cvx::Box, KdNode4> * worldTree() const;
	
protected:

private:

};

template<typename WorldT, typename InnerT, typename TreeT>
WorldManager<WorldT, InnerT, TreeT>::WorldManager() :
m_worldTree(0)
{}

template<typename WorldT, typename InnerT, typename TreeT>
WorldManager<WorldT, InnerT, TreeT>::~WorldManager()
{}

template<typename WorldT, typename InnerT, typename TreeT>
bool WorldManager<WorldT, InnerT, TreeT>::openWorld(const std::string & filename)
{
	bool stat = Manager<WorldT, InnerT, TreeT>::openWorld(filename);
	if(!stat) return stat;
	
	m_worldTree.reset(new WorldTreeT);
	m_worldTree->transfer(Manager<WorldT, InnerT, TreeT>::grid()->tree() );
	
	return stat;
}

template<typename WorldT, typename InnerT, typename TreeT>
CudaNTree<cvx::Box, KdNode4> * WorldManager<WorldT, InnerT, TreeT>::worldTree() const
{ return m_worldTree.get(); }

}

}

#endif


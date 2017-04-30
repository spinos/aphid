#ifndef APHID_JUL_MANAGER_H
#define APHID_JUL_MANAGER_H

/*
 *  Manager.h
 *  
 *
 *  Created by jian zhang on 4/6/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  world grid
 *  incore data and request queue
 *  limit memory usage
 */
#include <NTreeIO.h>
#include <boost/scoped_ptr.hpp>

namespace aphid {

namespace jul {

template<typename WorldT, typename InnerT, typename TreeT>
class Manager : public NTreeIO {
	
	boost::scoped_ptr<WorldT > m_grid;
	int m_usedMem;
	
public:
	Manager();
	virtual ~Manager();
	
	virtual bool openWorld(const std::string & filename);
	TreeT * loadCell(const int & idx);
	WorldT * grid() const;
	const int & usedMemory() const;
	
protected:
private:
};

template<typename WorldT, typename InnerT, typename TreeT>
Manager<WorldT, InnerT, TreeT>::Manager() :
m_grid(0),
m_usedMem(0)
{}

template<typename WorldT, typename InnerT, typename TreeT>
Manager<WorldT, InnerT, TreeT>::~Manager()
{ end(); }

template<typename WorldT, typename InnerT, typename TreeT>
bool Manager<WorldT, InnerT, TreeT>::openWorld(const std::string & filename)
{
	bool stat = false;
	
	if(!begin(filename) ) return false;
	
	stat = objectExists("/grid/.tree");
	
	if(stat) {
		m_grid.reset(new WorldT("/grid") );
		m_grid->load();
		m_usedMem += m_grid->tree()->usedMemory();
	
	} else {
		std::cout<<"\n  manager found no grid ";
	}
	
	return stat;
}

template<typename WorldT, typename InnerT, typename TreeT>
TreeT * Manager<WorldT, InnerT, TreeT>::loadCell(const int & idx)
{
	InnerT * cell = grid()->cell(idx);
	TreeT * r = cell->tree();
	if(r) return r;
	
	r = cell->loadTree();
	m_usedMem += r->usedMemory();
	return r;
}

template<typename WorldT, typename InnerT, typename TreeT>
WorldT * Manager<WorldT, InnerT, TreeT>::grid() const
{ return m_grid.get(); }

template<typename WorldT, typename InnerT, typename TreeT>
const int & Manager<WorldT, InnerT, TreeT>::usedMemory() const
{ return m_usedMem; }

}

}
#endif        //  #ifndef APHID_JUL_MANAGER_H

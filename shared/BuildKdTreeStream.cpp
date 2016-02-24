/*
 *  BuildKdTreeStream.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/29/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BuildKdTreeStream.h"

namespace aphid {

BuildKdTreeStream::BuildKdTreeStream() : m_numNodes(0) {}
BuildKdTreeStream::~BuildKdTreeStream() 
{ cleanup(); }

void BuildKdTreeStream::initialize()
{ cleanup(); }

void BuildKdTreeStream::cleanup()
{
	m_numNodes = 0;
	//std::vector<KdTreeNode *>::iterator it = m_nodeBlks.begin();
	//for(;it != m_nodeBlks.end(); ++it) free( *it);
	m_nodeBlks.clear();
	m_indirection.clear();
}

void BuildKdTreeStream::appendGeometry(Geometry * geo)
{
	const unsigned n = geo->numComponents();
	for(unsigned i = 0; i < n; i++) {
		Primitive p;
		p.setGeometry(geo);
		p.setComponentIndex(i);
		m_primitives.insert(p);
	}
}

const unsigned BuildKdTreeStream::getNumPrimitives() const
{
	return m_primitives.size();
}

const sdb::VectorArray<Primitive> &BuildKdTreeStream::getPrimitives() const
{
	return m_primitives;
}

sdb::VectorArray<Primitive> &BuildKdTreeStream::primitives()
{
	return m_primitives;
}

sdb::VectorArray<unsigned> &BuildKdTreeStream::indirection()
{
	return m_indirection;
}

KdTreeNode *BuildKdTreeStream::createTreeBranch()
{
    m_nodeBlks.insert();
    m_nodeBlks.insert();
    KdTreeNode * b = m_nodeBlks[m_numNodes];
/*	
	if((m_numNodes & 2047) == 0) {
		m_nodeBuf = (KdTreeNode *)malloc(sizeof(KdTreeNode) * 2050);
		m_nodeBlks.push_back(m_nodeBuf);
	}
	
	KdTreeNode * b = &m_nodeBuf[m_numNodes & 2047];	
*/	
	unsigned long * tmp = (unsigned long*)&b[0];
	tmp[1] = tmp[3] = 6;
	
	tmp = (unsigned long*)&b[1];
	tmp[1] = tmp[3] = 6;
	
	m_numNodes += 2;
	
	return b;
}

KdTreeNode *BuildKdTreeStream::firstTreeBranch()
{
	return m_nodeBlks[0];
}

const unsigned & BuildKdTreeStream::numNodes() const
{ return m_numNodes; }

void BuildKdTreeStream::verbose() const
{
    std::cout<<"\n kd-tree data stream input primitive count: " <<getNumPrimitives()
    <<"\n n node "<<m_numNodes
    <<"\n n blocks "<<m_nodeBlks.size();
}

}

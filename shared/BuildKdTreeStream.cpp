/*
 *  BuildKdTreeStream.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/29/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BuildKdTreeStream.h"

BuildKdTreeStream::BuildKdTreeStream() {}
BuildKdTreeStream::~BuildKdTreeStream() 
{
	cleanup();
}

void BuildKdTreeStream::initialize()
{}

void BuildKdTreeStream::cleanup()
{
	m_nodes.clear();
	m_indirection.clear();
}

void BuildKdTreeStream::appendGeometry(Geometry * geo)
{
// std::cout<<" geo type "<<geo->type()<<" ";
	const unsigned n = geo->numComponents();
	for(unsigned i = 0; i < n; i++) {
		//Primitive *p = m_primitives.asPrimitive();
		//p->setGeometry(geo);
		//p->setComponentIndex(i);
		//m_primitives.next();
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

std::vector<unsigned> &BuildKdTreeStream::indirection()
{
	return m_indirection;
}

KdTreeNode *BuildKdTreeStream::createTreeBranch()
{
	m_nodes.push_back(new KdTreeNode);
	KdTreeNode *p = m_nodes.back();
	unsigned long * tmp = (unsigned long*)p;
	tmp[1] = tmp[3] = 6;
	
	m_nodes.push_back(new KdTreeNode);
	KdTreeNode *q = m_nodes.back();
	tmp = (unsigned long*)q;
	tmp[1] = tmp[3] = 6;
	
	return p;
}

KdTreeNode *BuildKdTreeStream::firstTreeBranch()
{
	return m_nodes[0];
}

void BuildKdTreeStream::verbose() const
{
	printf("kd-tree data stream input primitive count: %i\nnodes state:\n", getNumPrimitives());	
}

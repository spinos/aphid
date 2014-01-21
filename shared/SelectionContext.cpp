/*
 *  SelectionContext.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/22/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "SelectionContext.h"
#include "QuickSort.h"

SelectionContext::SelectionContext() {}
SelectionContext::~SelectionContext() { m_indices.clear(); }

void SelectionContext::reset()
{
	m_indices.clear();
}

void SelectionContext::reset(const Vector3F & center, const float & radius)
{
	m_center = center;
	m_radius = radius;
	m_indices.clear();
}

void SelectionContext::setCenter(const Vector3F & center)
{
	m_center = center;
}

Vector3F SelectionContext::center() const
{
	return m_center;
}
	
void SelectionContext::setRadius(const float & radius)
{
	m_radius = radius;
}

float SelectionContext::radius() const
{
	return m_radius;
}

void SelectionContext::setBBox(const BoundingBox &bbox)
{
	m_bbox = bbox;
}

const BoundingBox & SelectionContext::getBBox() const
{
	return m_bbox;
}

char SelectionContext::closeTo(const BoundingBox & b) const
{
	return b.isPointAround(m_center, m_radius);
}

void SelectionContext::addToSelection(const unsigned idx)
{
	m_indices.push_back(idx);
}

void SelectionContext::finish()
{
	if(numSelected() < 2) return;
	QuickSort::Sort(m_indices, 0, numSelected() - 1);
	std::deque<unsigned>::iterator it = m_indices.begin();
	unsigned pre = *it;
	it++;
	for(; it != m_indices.end();) {
		if(*it == pre) {
			it = m_indices.erase(it);
		}
		else {
			pre = *it;
			it++;
		}
		
	}
}

unsigned SelectionContext::numSelected() const
{
	return m_indices.size();
}

const std::deque<unsigned> & SelectionContext::selectedQue() const
{
	return m_indices;
}

void SelectionContext::verbose() const
{
	std::clog<<"n selected: "<<numSelected()<<"\n";
	std::deque<unsigned>::const_iterator it = m_indices.begin();
	for(; it != m_indices.end(); ++it) {
		std::clog<<" "<<*it;
	}
}

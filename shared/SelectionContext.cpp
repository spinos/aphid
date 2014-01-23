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

SelectionContext::SelectionContext() { m_mode = Replace; }
SelectionContext::~SelectionContext() { m_indices.clear(); }

void SelectionContext::reset()
{
	if(m_mode == Replace) m_indices.clear();
}

void SelectionContext::reset(const Vector3F & center, const float & radius)
{
	m_center = center;
	m_radius = radius;
	if(m_mode == Replace) m_indices.clear();
	m_enableDirection = 0;
}

void SelectionContext::setDirection(const Vector3F & d)
{
	m_normal = d;
	m_enableDirection = 1;
}

void SelectionContext::setCenter(const Vector3F & center)
{
	m_center = center;
}

const Vector3F & SelectionContext::center() const
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

char SelectionContext::closeTo(const Vector3F & v) const
{
	if(!m_enableDirection) return 1;
	return m_normal.dot(v) > 0.1f;
}

void SelectionContext::addToSelection(const unsigned idx)
{
	if(m_mode == Replace || m_mode == Append) m_indices.push_back(idx);
	else m_removeIndices.push_back(idx);
}

void SelectionContext::finish()
{
	if(m_mode == Replace || m_mode == Append) finishAdd();
	else finishRemove();
}

void SelectionContext::finishAdd()
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

void SelectionContext::finishRemove()
{
	if(m_removeIndices.size() < 1) return;
	QuickSort::Sort(m_removeIndices, 0, m_removeIndices.size() - 1);
	std::deque<unsigned>::iterator it = m_removeIndices.begin();
	for(; it != m_removeIndices.end(); ++it) remove(*it);
	m_removeIndices.clear();
}

void SelectionContext::remove(const unsigned & idx)
{
	std::deque<unsigned>::iterator it = m_indices.begin();
	for(; it != m_indices.end();) {
		if(*it == idx) it = m_indices.erase(it);
		else if(*it > idx) return;
		else it++;
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

void SelectionContext::setSelectMode(SelectionContext::SelectMode m)
{
	m_mode = m;
}

SelectionContext::SelectMode SelectionContext::getSelectMode() const
{
	return m_mode;
}

void SelectionContext::verbose() const
{
	std::clog<<"n selected: "<<numSelected()<<"\n";
	std::deque<unsigned>::const_iterator it = m_indices.begin();
	for(; it != m_indices.end(); ++it) {
		std::clog<<" "<<*it;
	}
}

/*
 *  GeoElement.cpp
 *  convexHull
 *
 *  Created by jian zhang on 9/5/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "GeoElement.h"

GeoElement::GeoElement() : next(0)
{
	marked = 0;
	visibility = 1;
}

GeoElement::~GeoElement() {}

void GeoElement::setVisibility(char val)
{
	visibility = val;
}

char GeoElement::isVisible() const
{
	return visibility;
}

void GeoElement::setData(char *data)
{
	m_data = data;
}

char *GeoElement::getData()
{
	return m_data;
}

void GeoElement::setMarked(char val)
{
	marked = val;
}

char GeoElement::isMarked() const
{
	return marked;
}

void GeoElement::setIndex(int val)
{
	index = val;
}

int GeoElement::getIndex() const
{
	return index;
}

void GeoElement::setNext(GeoElement * another)
{
	next = another;
}

GeoElement* GeoElement::getNext() const
{
	return next;
}

/*
 *  MlFeatherCollection.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlFeatherCollection.h"

#include <MlFeather.h>

MlFeatherCollection::MlFeatherCollection()
{
}

MlFeatherCollection::~MlFeatherCollection()
{
}

void MlFeatherCollection::clearFeatherExamples()
{
	std::map<unsigned, MlFeather *>::iterator it;
	for(it = m_feathers.begin(); it != m_feathers.end(); ++it) {
		delete it->second;
	}
	m_feathers.clear();
}

MlFeather * MlFeatherCollection::addFeatherExample()
{
	MlFeather * f = new MlFeather;
	f->setFeatherId(numFeatherExamples());
	m_feathers[numFeatherExamples()] = f;
	return f;
}

unsigned MlFeatherCollection::numFeatherExamples() const
{
	return m_feathers.size();
}

void MlFeatherCollection::selectFeatherExample(unsigned x)
{
	m_selectedFeatherId = x;
}

MlFeather * MlFeatherCollection::selectedFeatherExample()
{
	return m_feathers[m_selectedFeatherId];
}

unsigned MlFeatherCollection::selectedFeatherExampleId() const
{
	return m_selectedFeatherId;
}

MlFeather * MlFeatherCollection::featherExample(unsigned idx)
{
	return m_feathers[idx];
}

void MlFeatherCollection::initializeFeatherExample()
{
    if(numFeatherExamples() < 1) {
		MlFeather * f = addFeatherExample();
		f->defaultCreate();
	}
	selectFeatherExample(0);
}

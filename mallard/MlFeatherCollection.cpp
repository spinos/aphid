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
	initializeFeatherExample();
}

MlFeatherCollection::~MlFeatherCollection()
{
}

void MlFeatherCollection::clearFeatherExamples()
{
	std::map<unsigned, MlFeather *>::iterator it;
	for(it = m_feathers.begin(); it != m_feathers.end(); ++it) {
		if(it->second) {
			delete it->second;
			it->second = 0;
		}
	}
	m_feathers.clear();
}

MlFeather * MlFeatherCollection::addFeatherExample()
{
	return addFeatherExampleId(usableId());
}

MlFeather * MlFeatherCollection::addFeatherExampleId(unsigned idx)
{
	MlFeather * f = new MlFeather;
	f->setFeatherId(idx);
	m_feathers[idx] = f;
	std::cout<<"add example["<<f->featherId()<<"]\n";
	return f;
}

bool MlFeatherCollection::selectFeatherExample(unsigned x)
{
	if(!featherIdExists(x)) return false;
		
	m_selectedFeatherId = x;
	return true;
}

MlFeather * MlFeatherCollection::selectedFeatherExample()
{
	return m_feathers[m_selectedFeatherId];
}

unsigned MlFeatherCollection::selectedFeatherExampleId() const
{
	return m_selectedFeatherId;
}

bool MlFeatherCollection::removeSelectedFeatherExample()
{
	if(numFeatherExamples() < 2) return false;
	std::map<unsigned, MlFeather *>::iterator it = m_feathers.find(m_selectedFeatherId);
	if(it == m_feathers.end()) return false;
	delete it->second;
	it->second = 0;
	return true;
}

MlFeather * MlFeatherCollection::featherExample(unsigned idx)
{	
	if(featherIdExists(idx)) 
		return m_feathers[idx];
	return 0;
}

void MlFeatherCollection::initializeFeatherExample()
{
    if(numFeatherExamples() < 1)
		addFeatherExample();
	selectFeatherExample(0);
}

unsigned MlFeatherCollection::numFeatherExamples() const
{
	unsigned nf = 0;
	std::map<unsigned, MlFeather *>::const_iterator it;
	for(it = m_feathers.begin(); it != m_feathers.end(); it++) {
		if(it->second)
			nf++;
	}
	
	return nf;
}

unsigned MlFeatherCollection::usableId() const
{
	std::map<unsigned, MlFeather *>::const_iterator it;
	for(it = m_feathers.begin(); it != m_feathers.end(); it++) {
		if(!(it->second))
			return it->first;
	}
	return numFeatherExamples();
}

bool MlFeatherCollection::featherIdExists(unsigned idx) const
{
	std::map<unsigned, MlFeather *>::const_iterator it = m_feathers.find(idx);
	if(it == m_feathers.end()) return false;
	if(it->second == 0) return false;
	return true;
}

MlFeather* MlFeatherCollection::firstFeatherExample()
{
	m_featherIt = m_feathers.begin();
	if(m_featherIt == m_feathers.end()) return 0;
	return m_featherIt->second;
}

MlFeather* MlFeatherCollection::nextFeatherExample()
{
	m_featherIt++;
	if(m_featherIt == m_feathers.end()) return 0;
	return m_featherIt->second;
}

bool MlFeatherCollection::hasFeatherExample()
{
	return m_featherIt != m_feathers.end();
}

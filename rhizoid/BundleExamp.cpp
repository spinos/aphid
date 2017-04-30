#include "BundleExamp.h"

namespace aphid {

BundleExamp::BundleExamp()
{}

BundleExamp::~BundleExamp()
{ 
    clearInstances();
	clearExamples(); 
}

void BundleExamp::clearExamples()
{ m_examples.clear(); }

void BundleExamp::clearInstances()
{ m_instances.clear(); }

void BundleExamp::addAExample(ExampVox * v)
{ m_examples.push_back(v); }

void BundleExamp::addAInstance(const InstanceD & v)
{ m_instances.push_back(v); } 

int BundleExamp::numExamples() const
{ return m_examples.size(); }

int BundleExamp::numInstances() const
{ return m_instances.size(); }

const ExampVox * BundleExamp::getExample(const int & i) const
{ return m_examples[i]; } 

ExampVox * BundleExamp::getExample(const int & i)
{ return m_examples[i]; }

const ExampVox::InstanceD & BundleExamp::getInstance(const int & i) const
{ return m_instances[i]; }

void BundleExamp::setActive(bool x)
{
	std::vector<ExampVox * >::iterator it = m_examples.begin();
	for(;it!=m_examples.end();++it) {
		(*it)->setActive(x);
	}
	ExampVox::setActive(x);
}

void BundleExamp::setVisible(bool x)
{
	std::vector<ExampVox * >::iterator it = m_examples.begin();
	for(;it!=m_examples.end();++it) {
		(*it)->setVisible(x);
	}
	ExampVox::setVisible(x);
}

}


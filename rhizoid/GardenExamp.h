/*
 *  GardenExamp.h
 *  rhizoid
 *
 *  Created by jian zhang on 5/12/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GARDEN_EXAMP_H
#define APH_GARDEN_EXAMP_H

#include "ExampVox.h"
#include <vector>

namespace aphid {

class CompoundExamp;

class GardenExamp : public ExampVox {
  
    std::vector<CompoundExamp * > m_examples;

public:
    GardenExamp();
    virtual ~GardenExamp();
	
	void addAExample(CompoundExamp * v);
    
    virtual int numExamples() const;
	virtual const ExampVox * getExample(const int & i) const;
	/*virtual int numInstances() const;
	virtual ExampVox * getExample(const int & i);
	virtual const InstanceD & getInstance(const int & i) const;
	
	void clearExamples();
	void clearInstances();
	
	virtual void setActive(bool x);
	virtual void setVisible(bool x);*/
	
protected:
    CompoundExamp * getCompoundExample(const int & i);
	
private:
    
};

}
#endif
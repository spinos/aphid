/*
 *  CompoundExamp.h
 *  rhizoid
 *
 *  multi-instanced
 *  Created by jian zhang on 5/12/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_COMPOUND_EXAMP_H
#define APH_COMPOUND_EXAMP_H

#include "ExampVox.h"
#include <vector>

namespace aphid {

class Matrix44F;

class CompoundExamp : public ExampVox {
  
    //std::vector<ExampVox * > m_examples;
    std::vector<InstanceD > m_instances;
	
public:
    CompoundExamp();
    virtual ~CompoundExamp();
	
	void addInstance(const Matrix44F & tm, const int & instanceId);
    
	virtual int numInstances() const;
    //virtual int numExamples() const;
	/*
	virtual const ExampVox * getExample(const int & i) const;
	virtual ExampVox * getExample(const int & i);
	virtual const InstanceD & getInstance(const int & i) const;
	
	void clearExamples();
	void clearInstances();
	
	void addAExample(ExampVox * v);
    
	virtual void setActive(bool x);
	virtual void setVisible(bool x);*/
	
protected:
    
private:
    
};

}
#endif
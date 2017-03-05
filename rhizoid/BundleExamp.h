#ifndef APH_BUNDLE_EXAMP_H
#define APH_BUNDLE_EXAMP_H

#include "ExampVox.h"
#include <vector>

namespace aphid {
 
class BundleExamp : public ExampVox {
  
    std::vector<ExampVox * > m_examples;
    std::vector<InstanceD > m_instances;
	
public:
    BundleExamp();
    virtual ~BundleExamp();
    
    virtual int numExamples() const;
	virtual int numInstances() const;
	virtual const ExampVox * getExample(const int & i) const;
	virtual ExampVox * getExample(const int & i);
	virtual const InstanceD & getInstance(const int & i) const;
	
	void clearExamples();
	void clearInstances();
	
	void addAExample(ExampVox * v);
    
	virtual void setActive(bool x);
	virtual void setVisible(bool x);
	
protected:
    void addAInstance(const InstanceD & v);
    
private:
    
};

}

#endif


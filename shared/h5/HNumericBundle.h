#ifndef APHID_H_NUMERIC_BUNDLE_H
#define APHID_H_NUMERIC_BUNDLE_H

#include <h5/HBase.h>

namespace aphid {

class ABundleAttribute;

class HNumericBundle : public HBase {
public:
	HNumericBundle(const std::string & path);
	virtual ~HNumericBundle();
	
	virtual char verifyType();
	virtual char save(const ABundleAttribute * d);
	virtual char load(ABundleAttribute * d);
	
protected:
	
private:
	
};

}
#endif        //  #ifndef HPOLYGONALMESH_H


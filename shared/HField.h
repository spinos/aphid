#ifndef HFIELD_H
#define HFIELD_H

#include <HBase.h>

class AField;

class HField : public HBase {
public:
	HField(const std::string & path);
	virtual ~HField();
	
	virtual char verifyType();
	virtual char save(AField * fld);
	virtual char load(AField * fld);

protected:
    
private:
	
};
#endif        //  #ifndef HFIELD_H


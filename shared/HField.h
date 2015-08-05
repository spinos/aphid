#ifndef HFIELD_H
#define HFIELD_H

#include <HBase.h>

class AField;
class TypedBuffer;
class HField : public HBase {
public:
	HField(const std::string & path);
	virtual ~HField();
	
	virtual char verifyType();
	virtual char save(AField * fld);
	virtual char load(AField * fld);

protected:
    void saveAChannel(const std::string& name, TypedBuffer * chan);
	void loadAChannel(const std::string& name, AField * fld);
private:
	
};
#endif        //  #ifndef HFIELD_H


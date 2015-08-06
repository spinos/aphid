#include "APlaybackFile.h"
#include <string>
#include <map>
class AField;
class H5FieldIn : public APlaybackFile {
public:
    H5FieldIn();
    H5FieldIn(const char * name);
    
    AField * fieldByName(const std::string & fieldName);

    virtual void addField(const std::string & fieldName,
                      AField * fld);
protected:
    const std::map<std::string, AField *> * fields() const;
private:
    std::map<std::string, AField *> m_fields;
};

#ifndef AVERBOSE_H
#define AVERBOSE_H

#include <string>
namespace aphid {

class AVerbose {
public:
    void verbose() const;
    virtual std::string verbosestr() const;
};

}
#endif        //  #ifndef AVERBOSE_H


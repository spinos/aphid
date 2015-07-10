#ifndef AVERBOSE_H
#define AVERBOSE_H

#include <string>

class AVerbose {
public:
    void verbose() const;
    virtual std::string verbosestr() const;
};
#endif        //  #ifndef AVERBOSE_H


#pragma once
#include "HFile.h"
#include "APlayback.h"
namespace aphid {

class APlaybackFile : public HFile, public APlayback {
public:
    APlaybackFile();
    APlaybackFile(const char * name);
    virtual ~APlaybackFile();
    
    bool writeFrameRange(AFrameRange * src);
    bool readFrameRange();
    
    virtual void verbose() const;
protected:

private:

};

}

#pragma once
#include "HFile.h"
#include "APlayback.h"

class APlaybackFile : public HFile, public APlayback {
public:
    APlaybackFile();
    APlaybackFile(const char * name);
    virtual ~APlaybackFile();
    
    bool writeFrameRange(AFrameRange * src);
    bool readFrameRange();
    
protected:

private:

};

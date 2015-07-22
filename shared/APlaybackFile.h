#pragma once
#include <HFile.h>
#include <AFrameRange.h>

class APlaybackFile : public HFile, public AFrameRange {
public:
    APlaybackFile();
    APlaybackFile(const char * name);
    
    bool isFrameBegin() const;
    bool isFrameEnd() const;
    void frameBegin();
    void nextFrame();
    const int currentFrame() const;
    
    bool readFrameRange();
protected:

private:
    int m_currentFrame;
};

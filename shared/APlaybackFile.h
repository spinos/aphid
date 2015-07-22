#pragma once
#include <HFile.h>
#include <AFrameRange.h>

class APlaybackFile : public HFile, public AFrameRange {
public:
    APlaybackFile();
    APlaybackFile(const char * name);
    virtual ~APlaybackFile();
    
    bool isFrameBegin() const;
    bool isFrameEnd() const;
    void frameBegin();
    void nextFrame();
    const int currentFrame() const;
    
    bool writeFrameRange(AFrameRange * src);
    bool readFrameRange();
    
    void beginCountNumFramesPlayed();
    void countNumFramesPlayed();
    const int numFramesPlayed() const;
    const bool allFramesPlayed() const;
protected:

private:
    int m_currentFrame;
    int m_numFramesPlayed;
};

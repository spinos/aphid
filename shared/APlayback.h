#ifndef APLAYBACK_H
#define APLAYBACK_H

#include "AFrameRange.h"
#include <string>

namespace aphid {

class APlayback : public AFrameRange {
public:
    APlayback();
    virtual ~APlayback();
    
    bool isFrameBegin() const;
    bool isFrameEnd() const;
    bool isOutOfRange() const;
    void frameBegin();
    void nextFrame();
    const int currentFrame() const;
	void setCurrentFrame(int x);
    
    void beginCountNumFramesPlayed();
    void countNumFramesPlayed();
    const int numFramesPlayed() const;
    const bool allFramesPlayed() const;
    
    void sampleBegin();
    void nextSample();
    const bool isSampleEnd() const;
    void begin();
    void next();
    const bool end() const;
    
protected:
    std::string currentFrameStr() const;
private:
    int m_currentFrame;
    int m_currentSample;
    int m_numFramesPlayed;
};

}
#endif        //  #ifndef APLAYBACK_H


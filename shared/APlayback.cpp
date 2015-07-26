#include "APlayback.h"

APlayback::APlayback() { m_numFramesPlayed = 0; }
APlayback::~APlayback() {}

bool APlayback::isFrameBegin() const
{ return m_currentFrame == FirstFrame; }

bool APlayback::isFrameEnd() const
{ return m_currentFrame == LastFrame; }

bool APlayback::isOutOfRange() const
{ return (m_currentFrame < FirstFrame ||
            m_currentFrame > LastFrame); }

void APlayback::frameBegin()
{ m_currentFrame = FirstFrame; }

void APlayback::nextFrame()
{ m_currentFrame++; }

const int APlayback::currentFrame() const
{ return m_currentFrame; }

void APlayback::beginCountNumFramesPlayed()
{ m_numFramesPlayed = 0; }

void APlayback::countNumFramesPlayed()
{ m_numFramesPlayed++; }

const int APlayback::numFramesPlayed() const
{ return m_numFramesPlayed; }

const bool APlayback::allFramesPlayed() const
{ return numFramesPlayed() >= numFramesInRange(); }

void APlayback::sampleBegin()
{ m_currentSample = 0; }
    
void APlayback::nextSample()
{ m_currentSample++; }

const bool APlayback::isSampleEnd() const
{ return (m_currentSample + 1) == SamplesPerFrame; }

void APlayback::begin()
{
    frameBegin();
    sampleBegin();
}

void APlayback::next()
{
    if(isSampleEnd()) {
        nextFrame();
        sampleBegin();
    }
    else 
        nextSample();
}

const bool APlayback::end() const
{ return (isFrameEnd() && isSampleEnd()); }
//:~

#ifndef ADENIUMRENDER_H
#define ADENIUMRENDER_H

class BaseBuffer;
class CUDABuffer;
class AdeniumRender {
public:
    AdeniumRender();
    virtual ~AdeniumRender();
    
    void initOnDevice();
    void resize(int w, int h);
    void reset();
private:
    bool isSizeValid(int x, int y) const;
    int numPixels() const;
    void * rgbz();
private:
    BaseBuffer * m_hostRgbz;
    CUDABuffer * m_deviceRgbz;
    int m_imageWidth, m_imageHeight;
    int m_initd;
};
#endif        //  #ifndef ADENIUMRENDER_H


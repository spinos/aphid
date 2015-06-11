#ifndef ADENIUMRENDER_H
#define ADENIUMRENDER_H

class BaseBuffer;
class CUDABuffer;
class BaseCamera;
class BvhTriangleSystem;
class AdeniumRender {
public:
    AdeniumRender();
    virtual ~AdeniumRender();
    
    void initOnDevice();
    bool resize(int w, int h);
    void reset();
	void setModelViewMatrix(float * src);
	void renderOrhographic(BaseCamera * camera, BvhTriangleSystem * tri);
	void renderPerspective(BaseCamera * camera);
	void sendToHost();
	
	const int imageWidth() const;
	const int imageHeight() const;
	void * hostRgbz();
	
	const bool isInitd() const;
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


#ifndef ADENIUMRENDER_H
#define ADENIUMRENDER_H

class BaseBuffer;
class CUDABuffer;
class CudaPixelBuffer;
class BaseCamera;
class BvhTriangleSystem;
class AdeniumRender {
public:
    AdeniumRender();
    virtual ~AdeniumRender();
    
    bool resize(int w, int h);
    void reset();
	void setModelViewMatrix(float * src);
	void renderOrhographic(BaseCamera * camera, BvhTriangleSystem * tri);
	void renderPerspective(BaseCamera * camera, BvhTriangleSystem * tri);
	void sendToHost();
	
	const int imageWidth() const;
	const int imageHeight() const;
	void bindBuffer();
	void unbindBuffer();

private:
    bool isSizeValid(int x, int y) const;
    int numPixels() const;
private:
    CudaPixelBuffer * m_deviceRgbaPix;
	CUDABuffer * m_depth;
    int m_imageWidth, m_imageHeight;
};
#endif        //  #ifndef ADENIUMRENDER_H


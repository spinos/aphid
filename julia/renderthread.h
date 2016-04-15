#ifndef RENDERTHREAD_H
#define RENDERTHREAD_H

#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>
#include <CudaRender.h>

QT_BEGIN_NAMESPACE
class QImage;
QT_END_NAMESPACE

class RenderThread : public QThread
{
    Q_OBJECT

public:
    RenderThread(QObject *parent = 0);
    ~RenderThread();

	void setR(aphid::CudaRender * r);
    void render(QSize resultSize);
	void tumble(int dx, int dy);
	void track(int dx, int dy);
	void zoom(int dz);

signals:
    void renderedImage(const QImage &image);

protected:
    void run();

private:
    QMutex mutex;
    QWaitCondition condition;

    QSize m_resultSize, m_portSize;
    bool restart;
    bool abort;
    
	aphid::CudaRender * m_r;
	
};

#endif

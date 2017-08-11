#ifndef RENDERTHREAD_H
#define RENDERTHREAD_H

#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>

QT_BEGIN_NAMESPACE
class QImage;
QT_END_NAMESPACE

class RenderInterface;

class RenderThread : public QThread
{
    Q_OBJECT

public:
    RenderThread(RenderInterface* interf, QObject *parent = 0);
    ~RenderThread();

    void render(double centerX, double centerY, double scaleFactor,
                QSize resultSize);
	
signals:
    void renderedImage();
	
protected:
    void run();

private:
    QMutex mutex;
    QWaitCondition condition;
	QSize m_resultSize;
    double centerX;
    double centerY;
    double scaleFactor;
    bool restart;
    bool abort;
	
	RenderInterface* m_interface;
	
};
//! [0]

#endif

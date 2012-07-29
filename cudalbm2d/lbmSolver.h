#ifndef RENDERTHREAD_H
#define RENDERTHREAD_H

#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>

QT_BEGIN_NAMESPACE
class QImage;
QT_END_NAMESPACE

class RenderThread : public QThread
{
    Q_OBJECT

public:
    RenderThread(QObject *parent = 0);
    ~RenderThread();
    
    int solverWidth() const;
    int solverHeight() const;

    void render();
	void addImpulse(int x, int y, float vx, float vy);
	void addObstacle(int x, int y);

signals:
    void renderedImage(const QImage &image, const unsigned &step);

protected:
    void run();

private:
    QMutex mutex;
    QWaitCondition condition;

    QSize resultSize;
    bool restart;
    bool abort;
	
	unsigned char *wall;

	void simulate();

	uchar *pixel;
	float *impulse;
};

#endif

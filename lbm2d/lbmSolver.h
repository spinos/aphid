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

    void render(int x, int y);

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
	
	float *ux;
	float *uy;
	short *map;
	float *density;
	float *lat[9];
	float tau;
	void simulate(char reset);
	void getMacro(int x, int y, float &rho, float &vx, float &vy);
	unsigned _step;
	uchar *pixel;
	void addImpulse(int x, int y);
	int impulse_x, impulse_y;
};

#endif

#ifndef LfThread_H
#define LfThread_H

#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>

#include "LfWorld.h"

QT_BEGIN_NAMESPACE
class QImage;
QT_END_NAMESPACE

class LfThread : public QThread
{
    Q_OBJECT

public:
    LfThread(LfWorld * world, QObject *parent = 0);
    virtual ~LfThread();

    void render(QSize resultSize);
	void initAtoms();

signals:
    void renderedImage(const QImage &image);

protected:
    void run();

private:
	LfWorld * m_world;
    QMutex mutex;
    QWaitCondition condition;

    QSize resultSize;
    bool restart;
    bool abort;

};

#endif

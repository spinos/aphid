#ifndef LfThread_H
#define LfThread_H

#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>

QT_BEGIN_NAMESPACE
class QImage;
QT_END_NAMESPACE

namespace lfr {

class LfWorld;

class LfThread : public QThread
{
    Q_OBJECT

public:
    LfThread(LfWorld * world, QObject *parent = 0);
    virtual ~LfThread();

    void render(QSize resultSize);
	void initAtoms();
	void beginLearn();
	
signals:
	void sendInitialDictionary(const QImage &image);
	void sendDictionary(const QImage &image);
	void sendSparsity(const QImage &image);
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

	QImage * m_spasityImg;
	QImage * m_dictImg;
};

}
#endif

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

class LfMachine;

class LfThread : public QThread
{
    Q_OBJECT

public:
    LfThread(LfMachine * world, QObject *parent = 0);
    virtual ~LfThread();

    void initAtoms();
	void beginLearn();
	
public slots:
	void endLearn();
	
signals:
	void sendInitialDictionary(const QImage &image);
	void sendDictionary(const QImage &image);
	void sendSparsity(const QImage &image);
    void sendPSNR(float value);
	void sendIterDone(int n);
	void sendCodedImage(const QImage &image);
	
protected:
    void run();

private:
	LfMachine * m_world;
    QMutex mutex;
    QWaitCondition condition;

    QSize resultSize;
    bool restart;
    bool abort;

	QImage * m_spasityImg;
	QImage * m_dictImg;
	QImage * m_codedImg;
	
};

}
#endif

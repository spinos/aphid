#ifndef LFR_IMAGE_DIALOG_H
#define LFR_IMAGE_DIALOG_H

#include <QImage>
#include <QDialog>

QT_BEGIN_NAMESPACE
class QLabel;
QT_END_NAMESPACE

namespace lfr {

class DictionaryView;
class LfMachine;
class ImageDialog : public QDialog
{
    Q_OBJECT

public:
    ImageDialog(LfMachine * world, QWidget *parent = 0);

public slots:
   void recvImage(const QImage &image);
   
signals:
	void sendImage(const QImage &image);
   
protected:
   
private:
	
private:
	DictionaryView * m_dictView;
	QLabel * m_statistics;
	
};

}
#endif
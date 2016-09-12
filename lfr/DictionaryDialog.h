#ifndef LFR_DICTIONARY_DIALOG_H
#define LFR_DICTIONARY_DIALOG_H

#include <QImage>
#include <QDialog>

QT_BEGIN_NAMESPACE
class QLabel;
QT_END_NAMESPACE

namespace lfr {

class DictionaryView;
class LfMachine;
class DictionaryDialog : public QDialog
{
    Q_OBJECT

public:
    DictionaryDialog(LfMachine * world, QWidget *parent = 0);

public slots:
   void recvDictionary(const QImage &image);
   
signals:
	void sendDictionary(const QImage &image);
   
protected:
   
private:
	
private:
	DictionaryView * m_dictView;
	QLabel * m_statistics;
	
};

}
#endif
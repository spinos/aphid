#ifndef RegexUi_H
#define RegexUi_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QTextEdit;
class QPushButton;
QT_END_NAMESPACE

//! [class definition]
class RegexUi : public QWidget
{
    Q_OBJECT

public:
    RegexUi(QWidget *parent = 0);
	
public slots:
	void doReMatch();
	void doReSearch();
	void doReReplace();

private:
	QTextEdit *contentLine;
	QLineEdit *expressionLine;
	QLineEdit *replaceLine;
	QTextEdit *resultLine;
    QPushButton *togglePushButton;
	
private:
	QStringList match(QString& content, QString& expression);
	QStringList search(QString& content, QString& expression);
};

#endif        //  #ifndef RegexUi_H


#include <QtGui>
#include "RegexUi.h"

RegexUi::RegexUi(QWidget *parent)
    : QWidget(parent)
{
    contentLine = new QTextEdit;
	expressionLine = new QLineEdit;
	resultLine = new QTextEdit;
	resultLine->setReadOnly(true);
    togglePushButton = new QPushButton(tr("Match"));
       QPushButton *searchButton = new QPushButton(tr("Search")); 

	connect(togglePushButton, SIGNAL(clicked()), this, SLOT(doReMatch()));
    connect(searchButton, SIGNAL(clicked()), this, SLOT(doReSearch()));
    

//! [layout]
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(contentLine);
	mainLayout->addWidget(expressionLine);
	mainLayout->addWidget(resultLine);
    mainLayout->addWidget(togglePushButton);
    mainLayout->addWidget(searchButton);
//! [layout]

//![setting the layout]    
    setLayout(mainLayout);
    setWindowTitle(tr("Regex Test"));
}

void RegexUi::doReMatch()
{
	QString content = contentLine->toPlainText();
	QString expression = expressionLine->text();
	QString log;
	QStringList res = match(content, expression);
	if(res.size() < 1)
		QTextStream(&log) << "no match!";
	else
		QTextStream(&log) << "matched:\n";
	QStringList::const_iterator constIterator;
     for (constIterator = res.constBegin(); constIterator != res.constEnd();
            ++constIterator)
	QTextStream(&log) << (*constIterator).toLocal8Bit().constData() << "\n";
	resultLine->setText(log);
}

void RegexUi::doReSearch()
{
	QString content = contentLine->toPlainText();
	QString expression = expressionLine->text();
	QString log;
	QStringList res = search(content, expression);
	if(res.size() < 1)
		QTextStream(&log) << "not found!";
	else
		QTextStream(&log) << "found:\n";
	QStringList::const_iterator constIterator;
     for (constIterator = res.constBegin(); constIterator != res.constEnd();
            ++constIterator)
	QTextStream(&log) << (*constIterator).toLocal8Bit().constData() << "\n";

	resultLine->setText(log);
}

#include <boost/regex.hpp>
#include <boost/format.hpp>

QStringList RegexUi::match(QString& content, QString& expression)
{
	QStringList res;
	char found = 0;
	const boost::regex re1(expression.toUtf8().data());
	
	std::string tomatch(content.toUtf8().data());
	std::string::const_iterator start, end;
    start = tomatch.begin();
    end = tomatch.end();
	boost::match_results<std::string::const_iterator> what;
	if(regex_match(tomatch, what, re1, boost::match_default) )
	{
		found = 1;
		for(unsigned i = 0; i <what.size(); i++)
		{
			std::string numblk = str(boost::format(" %1% : %2% ") % i % what[i]);
			res << numblk.c_str();
		}
	}
	return res;
}

QStringList RegexUi::search(QString& content, QString& expression)
{
	QStringList res;
	char found = 0;
	const boost::regex re1(expression.toUtf8().data());
	
	std::string tomatch(content.toUtf8().data());
	std::string::const_iterator start, end;
    start = tomatch.begin();
    end = tomatch.end();
	boost::match_results<std::string::const_iterator> what;
	while(regex_search(start, end, what, re1, boost::match_default) )
	{
		found = 1;
		for(unsigned i = 0; i <what.size(); i++)
		{
			std::string numblk = str(boost::format(" %1% : %2% ") % i % what[i]);
			res << numblk.c_str();
		}
		start = what[0].second;
	}
	return res;
}

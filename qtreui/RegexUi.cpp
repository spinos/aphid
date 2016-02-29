#include <QtGui>
#include "RegexUi.h"
#include "Reagan.h"
RegexUi::RegexUi(QWidget *parent)
    : QWidget(parent)
{
    contentLine = new QTextEdit("-100/0/4;300/456/2;99/199/5;");
	expressionLine = new QLineEdit("(?<0>-*[[:digit:]]+)/(?<1>-*[[:digit:]]+)/(?<2>-*[[:digit:]]+);");
	replaceLine = new QLineEdit("b");
	resultLine = new QTextEdit;
	resultLine->setReadOnly(true);
    togglePushButton = new QPushButton(tr("Match"));
       QPushButton *searchButton = new QPushButton(tr("Search")); 
	QPushButton *replaceButton = new QPushButton(tr("Replace")); 


	connect(togglePushButton, SIGNAL(clicked()), this, SLOT(doReMatch()));
    connect(searchButton, SIGNAL(clicked()), this, SLOT(doReSearch()));
    connect(replaceButton, SIGNAL(clicked()), this, SLOT(doReReplace()));
    

//! [layout]
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(contentLine);
	mainLayout->addWidget(expressionLine);
	mainLayout->addWidget(replaceLine);
	mainLayout->addWidget(resultLine);
	QHBoxLayout *buttonGrp = new QHBoxLayout;
    
    buttonGrp->addWidget(togglePushButton);
    buttonGrp->addWidget(searchButton);
	buttonGrp->addWidget(replaceButton);
	mainLayout->addLayout(buttonGrp);
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

void RegexUi::doReReplace()
{
	std::string b4(contentLine->toPlainText().toUtf8().data());
	std::string aft = b4;
	Reagan::runReReplace(aft, expressionLine->text().toUtf8().data(), replaceLine->text().toUtf8().data());
	QString log;
	
	if(b4 == aft)
		QTextStream(&log) << "not replaced!";
	else
		QTextStream(&log) << "found and replaced to:\n" << aft.c_str();
	
	resultLine->setText(log);
	
	std::string unixpathname("\\\\storage.company.some\\slow/stage\\share///file.mm");
	Reagan::validateUnixPath(unixpathname);
	
	qDebug() << unixpathname.c_str();
	
	std::string namest("|abc:a_v0|abc:b_b2|abc:cc_grp");
	Reagan::removeNamespaceInFullPathName(namest);
	
	qDebug() << namest.c_str();
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
	int j=0;
	std::string tomatch(content.toUtf8().data());
	std::string::const_iterator start, end;
    start = tomatch.begin();
    end = tomatch.end();
	boost::match_results<std::string::const_iterator> what;
	while(regex_search(start, end, what, re1, boost::match_default) )
	{
        res <<str( boost::format(" occurance[%1%] ") % (j++) ).c_str();
		found = 1;
		for(unsigned i = 0; i <what.size(); ++i)
		{
			std::string numblk = str(boost::format(" %1% : %2% ") % i % what[i]);
			res << numblk.c_str();
		}
		start = what[0].second;
	}
	return res;
}


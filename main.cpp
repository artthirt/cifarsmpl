#include "mainwindow.h"
#include <QApplication>

#include <QDir>

#include "test_agg.h"

int main(int argc, char *argv[])
{
	QString progpath = argv[0];
	QDir dir;
	dir.setPath(progpath);
	dir.cd("../");
	QDir::current().setCurrent(dir.canonicalPath());

	test_agg tagg;
	tagg.test_hconcat();

	QApplication a(argc, argv);
	MainWindow w;
	w.show();

	return a.exec();
}

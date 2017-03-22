#include "mainwindow.h"
#include <QApplication>

#include <QDir>

#include "test_agg.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	QString progpath = argv[0];
	QDir dir;
	dir.setPath(progpath);
	dir.cd("../");
	QDir::current().setCurrent(dir.canonicalPath());
#if 0
	test_agg tagg;
//	if(1)
//		tagg.test_file();
//	else
//		tagg.test_conv_gpu();
//	tagg.test_hconcat();
//	tagg.test_im2col();
//	tagg.test_conv();
//	tagg.test_norm();
	tagg.test_back();

	return 0;
#endif

	MainWindow w;
	w.show();

	return a.exec();
}

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QSettings>

#include "cifar_reader.h"

const QString model_file("model.bin");

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	m_doIter = false;
	m_batch = 50;
	m_delimiter = 50;
	m_delay = 5;
	m_wid = 0;

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(m_delay);

	ui->wcifar->setCifar(&m_cifar);

	std::vector< ct::ParamsCnv > cnv;
	std::vector< ct::ParamsMlp > mlp;

	ct::generator.seed(17);

	cnv.push_back(ct::ParamsCnv(3, 64, false, 1, 0, 2));
	cnv.push_back(ct::ParamsCnv(3, 128, false, 1, 0, 2));
	cnv.push_back(ct::ParamsCnv(3, 256, false, 1, 0));
//	cnv.push_back(ct::ParamsCnv(1, 128, false, 1, 0));
	cnv.push_back(ct::ParamsCnv(3, 512, false, 1, 0));

//	mlp.push_back(ct::ParamsMlp(512, 0.9, 0.001));
	mlp.push_back(ct::ParamsMlp(512, 0.85, 0.0001));
	mlp.push_back(ct::ParamsMlp(512, 0.85, 0.0001));
//	mlp.push_back(ct::ParamsMlp(512, 0.95, 0.001));
//	mlp.push_back(ct::ParamsMlp(512, 0.98, 0.0));
//	mlp.push_back(ct::ParamsMlp(128, 1));
	mlp.push_back(ct::ParamsMlp(10, 1));

	m_train.setCifar(&m_cifar);
	m_train.setConvLayers(cnv, ct::Size(32, 32));
	m_train.setMlpLayers(mlp);

	m_train.setUseRandData(true);
	m_train.setRandData(5, 0, 0.1);

	m_train.init();

	ui->tb_main->setCurrentIndex(0);

	ui->pte_logs->appendPlainText("Current directory: " + m_cifar.currentDirectory());
	ui->pte_logs->appendPlainText("Binary data exists: " + QString(m_cifar.isBinDataExists()? "True" : "False"));

	ui->pte_logs->appendPlainText(QString("count of input to MLP %1").arg(m_train.inputToMlp()));

	ui->sb_batch->setValue(m_batch);
	ui->sb_delay->setValue(m_delay);
	ui->sb_iter_numb->setValue(m_delimiter);

//	ui->dsb_dropoutprob->setValue(m_train.dropoutProb());

	ui->sb_wid->setMaximum((int)cnv.size());

	load_settings();
}

MainWindow::~MainWindow()
{
	save_settings();

	delete ui;
}

void MainWindow::on_pb_prev_clicked()
{
	ui->wcifar->prev();
}

void MainWindow::on_pb_next_clicked()
{
	ui->wcifar->next();
}

void MainWindow::on_pb_test_clicked()
{
	if(!m_cifar.isBinDataExists()){
		ui->pte_logs->appendPlainText("<<<Not exists binary data!>>>");
		return;
	}

	if(ui->pb_pass->isChecked()){
		ui->pte_logs->appendPlainText("<<<Not run test when pass work!>>>");
		return;
	}
	double acc = -1, l2 = -1;
	m_train.getEstimateTest(acc, l2, ui->chb_gpu->isChecked());

	ui->lb_out->setText(QString("Test: acc=%1, l2=%2").arg(acc).arg(l2));
	ui->pte_logs->appendPlainText(QString("Test: acc=%1, l2=%2").arg(acc).arg(l2));

	update_statistics();
}

void MainWindow::onTimeout()
{
	if(m_doIter){
		pass();
	}
}

void MainWindow::on_pb_pass_clicked()
{

}

void MainWindow::pass()
{
	double l2, acc;

	m_train.pass(m_batch, ui->chb_gpu->isChecked());

	uint it = ui->chb_gpu->isChecked()? m_train.iteration_gpu() : m_train.iteration();

	ui->lb_it->setText(QString("Iteraion %1").arg(it));

	if((it % m_delimiter) == 0){
		if(ui->wcifar->mode() == WidgetCifar::TRAIN)
			m_train.getEstimate(800, acc, l2, ui->chb_gpu->isChecked());
		else{
			m_train.getEstimateTest(800, acc, l2, ui->chb_gpu->isChecked());
		}

		qDebug("iteration %d: acc=%f, l2=%f", it, acc, l2);

		ui->lb_out->setText(QString("iteration %1: acc=%2, l2=%3").arg(it).arg(acc).arg(l2));
		ui->pte_logs->appendPlainText(QString("iteration %1: acc=%2, l2=%3").arg(it).arg(acc).arg(l2));

		update_prediction();
		update_statistics();
	}
}

void MainWindow::update_prediction()
{
	QVector<int> pr = m_train.predict(ui->wcifar->output_data(), ui->chb_gpu->isChecked());

	ui->wcifar->updatePredictfromIndex(pr);

	bool b = ui->chb_gpu->isChecked();
	ui->wdgW->setMat(m_train.cnvW(m_wid, b),
					 m_train.szW(m_wid, b),
					 m_train.Kernels(m_wid, b),
					 m_train.channels(m_wid, b));
}

void MainWindow::update_statistics()
{
	QString stat ="<b>Statistics:</b><br>";
	stat += "<tr> <td><b>y/p</b></td>\t";
	for(int i = 0; i < 10; ++i){
		stat += "<td><b>" + QString::number(i + 1) + "</b></td>";
	}
	stat += "</tr>";
	for(int i = 0; i < 10; ++i){
		stat += "<tr><td><b>" + QString::number(i + 1) + ":</b></td>";
		for(int j = 0; j < 10; ++j){
			double p = m_train.statistics(i, j);
			if(i == j){
				stat += "<td><b>" + QString::number(p, 'f', 2) + "</b></td>";
			}else{
				stat += "<td>" + QString::number(p, 'f', 2) + "</td>";
			}
		}
		stat += "</tr>";
	}
	ui->lb_stat->setText(stat);
}

void MainWindow::load_settings()
{
	QSettings settings("config.main.ini", QSettings::IniFormat);
	settings.beginGroup("main");
	int tmp = settings.value("batch").toInt();
	ui->sb_batch->setValue(tmp);

	ui->dsb_alpha->setValue(settings.value("alpha").toDouble());

	ui->sb_delay->setValue(settings.value("timeout").toInt());

	ui->sb_iter_numb->setValue(settings.value("num_pass_check").toInt());

	settings.endGroup();

}

void MainWindow::save_settings()
{
	QSettings settings("config.main.ini", QSettings::IniFormat);
	settings.beginGroup("main");
	settings.setValue("batch", ui->sb_batch->value());
	settings.setValue("alpha", ui->dsb_alpha->value());
	settings.setValue("timeout", ui->sb_delay->value());
	settings.setValue("num_pass_check", ui->sb_iter_numb->value());
	settings.endGroup();
}

void MainWindow::open_file(const QString &fn)
{
	QImage image;
	image.load(fn);
	if(image.isNull())
		return;

	QImage im256, im32;

	im256 = image.scaled(QSize(256, 256));
	im32 = image.scaled(QSize(32, 32));
	im32 = im32.convertToFormat(QImage::Format_RGB32);

	ui->lb_image->setPixmap(QPixmap::fromImage(im256));

	std::vector< ct::Matf > vX;
	ct::Matf A, X;

	QByteArray ba;
	ba.resize(32 * 32 * 3);

	unsigned char* x1 = (unsigned char*)ba.data();
	unsigned char* x2 = (unsigned char*)&ba.data()[1 * 32 * 32];
	unsigned char* x3 = (unsigned char*)&ba.data()[2 * 32 * 32];

	for(int y = 0; y < 32; ++y){
		QRgb* sc = (QRgb*)im32.scanLine(y);
		for(int x = 0; x < 32; ++x){
			x1[y * 32 + x] = qRed(sc[x]);
			x2[y * 32 + x] = qGreen(sc[x]);
			x3[y * 32 + x] = qBlue(sc[x]);
		}
	}

	ct::image2mat<float>(ba, 32, 32, X);
	vX.push_back(X);

	m_train.forward(vX, A, false, ui->chb_gpu->isChecked());

	if(A.empty())
		return;

	float* dA = A.ptr();

	QString metaNames[] = {
		"airplane",
		"automobile",
		"bird",
		"cat",
		"deer",
		"dog",
		"frog",
		"horse",
		"ship",
		"truck"
	};

	QString out = "Probablity: ";

	struct tmeta{
		float prob;
		int index;
	};

	std::vector< tmeta > meta;
	meta.resize(10);

	for(int i = 0; i < 10; ++i){
		meta[i].prob = dA[i];
		meta[i].index = i;
//		out += "\n" + QString::number(i + 1) + ": " + QString::number(dA[i], 'f', 3);
	}

	qSort(meta.begin(), meta.end(), [&](const tmeta& t1, const tmeta& t2) {return t1.prob > t2.prob;});

	for(size_t i = 0; i < meta.size(); ++i){
		out += "\nprob[" + QString::number(meta[i].prob, 'f', 3) + "]: " + QString(metaNames[meta[i].index]);
	}

	ui->lb_prob->setText(out);
}

void MainWindow::on_pb_pass_clicked(bool checked)
{
	if(!m_cifar.isBinDataExists()){
		checked = false;
		ui->pb_pass->setChecked(false);
	}
	m_doIter = checked;
}

void MainWindow::on_chb_gpu_clicked(bool checked)
{
	if(checked && m_cifar.isBinDataExists()){
		m_train.init_gpu();
		ui->pte_logs->appendPlainText(QString("GPU: count of input to MLP %1").arg(m_train.inputToMlp(true)));
	}
}

void MainWindow::on_dsb_alpha_valueChanged(double arg1)
{
	m_train.setAlpha(arg1);
	m_train.setAlphaCnv(arg1);
}

void MainWindow::on_actOpenDir_triggered()
{
	QFileDialog dlg;
	dlg.setAcceptMode(QFileDialog::AcceptOpen);
	dlg.setFileMode(QFileDialog::Directory);

	if(dlg.exec()){
		if(m_cifar.openDir(dlg.directory().absolutePath())){
			ui->wcifar->update_source();
		}
	}
}

void MainWindow::on_action_save_model_triggered()
{
	m_train.saveToFile(model_file, ui->chb_gpu->isChecked());
	ui->pte_logs->appendPlainText("Model saved");
}

void MainWindow::on_actionLoad_model_triggered()
{
	if(!m_train.loadFromFile(model_file, ui->chb_gpu->isChecked())){
		ui->pte_logs->appendPlainText("Model not loaded");
	}
	ui->pte_logs->appendPlainText("Model loaded");
}

void MainWindow::on_sb_delay_valueChanged(int arg1)
{
	m_delay = arg1;
	m_timer.setInterval(arg1);
}

void MainWindow::on_sb_batch_valueChanged(int arg1)
{
	m_batch = arg1;
}

void MainWindow::on_sb_iter_numb_valueChanged(int arg1)
{
	m_delimiter = arg1;
}

void MainWindow::on_pb_mode_clicked(bool checked)
{
	if(checked)
		ui->wcifar->setTestMode();
	else
		ui->wcifar->setTrainMode();
}

void MainWindow::on_pb_update_clicked()
{
	if(!m_cifar.isBinDataExists()){
		return;
	}

	update_prediction();
	update_statistics();
}

void MainWindow::on_sb_wid_valueChanged(int arg1)
{
	m_wid = arg1;

	bool b = ui->chb_gpu->isChecked();
	ui->wdgW->setMat(m_train.cnvW(m_wid, b),
					 m_train.szW(m_wid, b),
					 m_train.Kernels(m_wid, b),
					 m_train.channels(m_wid, b));
}

void MainWindow::on_dsb_dropoutprob_valueChanged(double arg1)
{
	m_train.setDropoutProb(arg1);
}

void MainWindow::on_actionSave_matrices_triggered()
{
	m_train.save_weights(ui->chb_gpu->isChecked());
}

void MainWindow::on_chb_debug_clicked(bool checked)
{
	m_train.setDebug(checked);
}

void MainWindow::on_dsb_lambda_valueChanged(double arg1)
{
	m_train.setLambdaMlp(arg1);
}

void MainWindow::on_actionOpen_image_triggered()
{
	QStringList sl;
	sl << "*.bmp *.jpg *.jpeg *.bmp *.png *.tif *.gif";

	QFileDialog dlg;
	dlg.setAcceptMode(QFileDialog::AcceptOpen);
	dlg.setNameFilters(sl);

	if(dlg.exec()){
		open_file(dlg.selectedFiles()[0]);
	}
}

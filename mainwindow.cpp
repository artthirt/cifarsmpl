#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>

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

	connect(&m_timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
	m_timer.start(m_delay);

	ui->wcifar->setCifar(&m_cifar);

	std::vector< int > cnv;
	std::vector< int > mlp;
	std::vector< int > cnv_w;
	std::vector< bool > cnv_p;

	cnv.push_back(10);
	cnv.push_back(5);
	cnv.push_back(3);

	cnv_w.push_back(5);
	cnv_w.push_back(5);
	cnv_w.push_back(5);
	cnv_p.push_back(false);
	cnv_p.push_back(true);
	cnv_p.push_back(true);

//	mlp.push_back(1000);
//	mlp.push_back(900);
	mlp.push_back(800);
	mlp.push_back(500);
	mlp.push_back(10);

	m_train.setCifar(&m_cifar);
	m_train.setConvLayers(cnv, cnv_w, ct::Size(32, 32), &cnv_p);
	m_train.setMlpLayers(mlp);

	m_train.setUseRandData(true);
	m_train.setRandData(2, 2);

	m_train.init();

	ui->tb_main->setCurrentIndex(0);

	ui->pte_logs->appendPlainText("Current directory: " + m_cifar.currentDirectory());
	ui->pte_logs->appendPlainText("Binary data exists: " + QString(m_cifar.isBinDataExists()? "True" : "False"));

	ui->sb_batch->setValue(m_batch);
	ui->sb_delay->setValue(m_delay);
	ui->sb_iter_numb->setValue(m_delimiter);
}

MainWindow::~MainWindow()
{
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

	std::vector< double > percents;

	m_train.pass(m_batch, ui->chb_gpu->isChecked(), &percents);

	{
		std::stringstream ss;
		ss << std::setprecision(2);
		for(size_t i = 0; i < percents.size(); ++i){
			ss << percents[i] << " ";
		}
		//ui->pte_logs->appendPlainText(QString(">> ") + ss.str().c_str());
	}

	uint it = ui->chb_gpu->isChecked()? m_train.iteration_gpu() : m_train.iteration();

	ui->lb_it->setText(QString("Iteraion %1").arg(it));

	if((it % m_delimiter) == 0){
		m_train.getEstimate(800, acc, l2, ui->chb_gpu->isChecked());

		qDebug("iteration %d: acc=%f, l2=%f", it, acc, l2);

		ui->lb_out->setText(QString("iteration %1: acc=%2, l2=%3").arg(it).arg(acc).arg(l2));
		ui->pte_logs->appendPlainText(QString("iteration %1: acc=%2, l2=%3").arg(it).arg(acc).arg(l2));

		update_prediction();
	}
}

void MainWindow::update_prediction()
{
	QVector<int> pr = m_train.predict(ui->wcifar->output_data(), ui->chb_gpu->isChecked());

	ui->wcifar->updatePredictfromIndex(pr);

	ui->wdgW->set_weightR(m_train.cnvW(0, ui->chb_gpu->isChecked()));
	ui->wdgW->set_weightG(m_train.cnvW(1, ui->chb_gpu->isChecked()));
	ui->wdgW->set_weightB(m_train.cnvW(2, ui->chb_gpu->isChecked()));
	ui->wdgW->update();
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
	if(checked && m_cifar.isBinDataExists())
		m_train.init_gpu();
}

void MainWindow::on_dsb_alpha_valueChanged(double arg1)
{
	m_train.setAlpha(arg1);
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

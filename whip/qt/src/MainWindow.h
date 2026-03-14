#pragma once

#include <QMainWindow>
#include <QTimer>

class QDoubleSpinBox;
class QSpinBox;
class QCheckBox;
class QPushButton;
class QLabel;
class ChainSimulator;
class ChainWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private slots:
    void tick();
    void onStartReset();
    void onRecord();

private:
    void buildAndStart(bool recording);
    void stopSim();
    void setParamsEnabled(bool on);
    void exportRecording();

    ChainSimulator *m_sim;
    ChainWidget *m_view;
    QTimer *m_timer;

    // State
    bool m_running = false;
    double m_simTime = 0.0;
    int m_stepCount = 0;

    // Parameter widgets
    QSpinBox       *m_spinBeads;
    QDoubleSpinBox *m_spinLength;
    QDoubleSpinBox *m_spinMass;
    QDoubleSpinBox *m_spinStiffness;
    QDoubleSpinBox *m_spinDamping;
    QDoubleSpinBox *m_spinDrag;
    QDoubleSpinBox *m_spinTaper;
    QCheckBox      *m_chkConstraints;
    QDoubleSpinBox *m_spinGravity;
    QDoubleSpinBox *m_spinDt;
    QSpinBox       *m_spinSubSteps;
    QSpinBox       *m_spinRecordInterval;

    QPushButton *m_btnStartReset;
    QPushButton *m_btnRecord;
    QLabel      *m_lblFrames;
};

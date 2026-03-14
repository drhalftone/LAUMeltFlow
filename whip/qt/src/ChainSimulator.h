#pragma once

#include <QObject>
#include <QVector>
#include <QVector2D>

class Bead;

/// Describes one rod element (edge) between two beads.
struct Rod {
    int beadA;
    int beadB;
    double restLength;
};

/// One snapshot of the full chain state at a single timestep.
struct FrameRecord {
    double time;
    QVector<QVector2D> positions;
    QVector<QVector2D> velocities;
};

/// Coordinator that owns all beads and rods, wires up signals/slots,
/// and drives the two-phase timestep:
///   Phase 1: broadcast neighbor states -> beads accumulate forces
///   Phase 2: all beads integrate simultaneously
///   Phase 3 (optional): SHAKE constraint projection
///
class ChainSimulator : public QObject {
    Q_OBJECT

public:
    explicit ChainSimulator(QObject *parent = nullptr);

    /// Build a horizontal chain of n beads, fixed at the left end.
    /// taperRatio: mass ratio base/tip (1.0 = uniform, >1 = whip taper).
    void buildChain(int nBeads, double totalLength, double totalMass,
                    double stiffness, double damping, double drag = 0.0,
                    QVector2D anchorPos = QVector2D(0, 0),
                    double taperRatio = 1.0);

    /// Advance the simulation by one timestep.
    void step(double dt, double gravity);

    /// Enable/disable SHAKE constraint projection for inextensible rods.
    void setUseConstraints(bool on) { m_useConstraints = on; }
    bool useConstraints() const { return m_useConstraints; }
    void setConstraintIters(int n) { m_constraintIters = n; }
    int constraintIters() const { return m_constraintIters; }

    // Recording
    void setRecording(bool on) { m_recording = on; }
    bool isRecording() const { return m_recording; }
    void recordFrame(double time);
    const QVector<FrameRecord> &recording() const { return m_frames; }
    void clearRecording() { m_frames.clear(); }

    // Accessors for the renderer
    int beadCount() const { return m_beads.size(); }
    const Bead *bead(int i) const { return m_beads[i]; }
    const QVector<Rod> &rods() const { return m_rods; }

signals:
    /// Emitted after each step so the widget can repaint.
    void stepped();

private:
    void projectConstraints();

    QVector<Bead *> m_beads;
    QVector<Rod> m_rods;

    bool m_useConstraints = false;
    int m_constraintIters = 10;

    bool m_recording = false;
    QVector<FrameRecord> m_frames;
};

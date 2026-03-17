#include "ChainSimulator.h"
#include "Bead.h"

#include <cmath>

ChainSimulator::ChainSimulator(QObject *parent)
    : QObject(parent)
{
}

void ChainSimulator::buildChain(int nBeads, double totalLength,
                                double totalMass, double stiffness,
                                double damping, double drag,
                                QVector2D anchorPos, double taperRatio)
{
    // Clean up any previous chain
    qDeleteAll(m_beads);
    m_beads.clear();
    m_rods.clear();
    m_frames.clear();

    double L0 = totalLength / (nBeads - 1);

    // Compute per-bead mass with linear taper
    QVector<double> masses(nBeads);
    if (taperRatio <= 1.0) {
        double massPerBead = totalMass / nBeads;
        for (int i = 0; i < nBeads; ++i)
            masses[i] = massPerBead;
    } else {
        // Linear taper: t[i] goes from 1.0 (base) to 1/taperRatio (tip)
        double sum = 0.0;
        for (int i = 0; i < nBeads; ++i) {
            double t = 1.0 - (double)i / (nBeads - 1) * (1.0 - 1.0 / taperRatio);
            masses[i] = t;
            sum += t;
        }
        for (int i = 0; i < nBeads; ++i)
            masses[i] = masses[i] / sum * totalMass;
    }

    // Create beads
    for (int i = 0; i < nBeads; ++i) {
        auto *bead = new Bead(i, masses[i], stiffness, damping, drag, this);
        bead->setPos(anchorPos + QVector2D(static_cast<float>(i * L0), 0));
        bead->setVel(QVector2D(0, 0));
        if (i == 0)
            bead->setFixed(true);
        m_beads.append(bead);
    }

    // Create rod elements
    for (int i = 0; i < nBeads - 1; ++i) {
        Rod rod;
        rod.beadA = i;
        rod.beadB = i + 1;
        rod.restLength = L0;
        m_rods.append(rod);
    }
}

void ChainSimulator::step(double dt, double gravity)
{
    const int n = m_beads.size();

    // -- Capture pre-step state for training data --
    QVector<QVector2D> prePos, preVel;
    if (m_recordBeadSamples) {
        prePos.resize(n);
        preVel.resize(n);
        for (int i = 0; i < n; ++i) {
            prePos[i] = m_beads[i]->pos();
            preVel[i] = m_beads[i]->vel();
        }
    }

    // -- Phase 1: Force accumulation (message passing) --

    for (Bead *b : m_beads)
        b->resetForces();

    for (Bead *b : m_beads)
        b->applyGravity(gravity);

    for (const Rod &rod : m_rods) {
        Bead *a = m_beads[rod.beadA];
        Bead *b = m_beads[rod.beadB];

        b->onNeighborState(a->id(), a->pos(), a->vel(), rod.restLength);
        a->onNeighborState(b->id(), b->pos(), b->vel(), rod.restLength);
    }

    // -- Phase 2: Integration (node update) --

    for (Bead *b : m_beads)
        b->integrate(dt);

    // -- Phase 3: Constraint projection (optional) --

    if (m_useConstraints)
        projectConstraints();

    // -- Capture post-step state and build training samples --
    if (m_recordBeadSamples) {
        for (int i = 0; i < n; ++i) {
            BeadSample s;
            s.beadId = i;
            s.posBefore = prePos[i];
            s.velBefore = preVel[i];
            s.mass = m_beads[i]->mass();
            s.fixed = m_beads[i]->isFixed();

            // Left neighbor (i-1)
            if (i > 0) {
                s.hasLeft = true;
                s.leftPos = prePos[i - 1];
                s.leftVel = preVel[i - 1];
                s.leftMass = m_beads[i - 1]->mass();
                s.leftRestLength = m_rods[i - 1].restLength;
            } else {
                s.hasLeft = false;
                s.leftPos = prePos[i];  // ghost: same position
                s.leftVel = QVector2D(0, 0);
                s.leftMass = 0.0;
                s.leftRestLength = 0.0;
            }

            // Right neighbor (i+1)
            if (i < n - 1) {
                s.hasRight = true;
                s.rightPos = prePos[i + 1];
                s.rightVel = preVel[i + 1];
                s.rightMass = m_beads[i + 1]->mass();
                s.rightRestLength = m_rods[i].restLength;
            } else {
                s.hasRight = false;
                s.rightPos = prePos[i];  // ghost: same position
                s.rightVel = QVector2D(0, 0);
                s.rightMass = 0.0;
                s.rightRestLength = 0.0;
            }

            s.posAfter = m_beads[i]->pos();
            s.velAfter = m_beads[i]->vel();

            m_beadSamples.append(s);
        }
    }

    emit stepped();
}

void ChainSimulator::projectConstraints()
{
    // SHAKE-like iterative projection to enforce rod inextensibility.
    // Adjusts positions along each rod so |r_j - r_i| = L0,
    // then corrects velocities to stay consistent.

    for (int iter = 0; iter < m_constraintIters; ++iter) {
        double maxErr = 0.0;

        for (const Rod &rod : m_rods) {
            Bead *a = m_beads[rod.beadA];
            Bead *b = m_beads[rod.beadB];

            QVector2D delta = b->pos() - a->pos();
            float dist = delta.length();
            if (dist < 1e-7f)
                continue;

            float err = dist - static_cast<float>(rod.restLength);
            if (std::abs(err) > maxErr)
                maxErr = std::abs(err);

            QVector2D dir = delta / dist;

            double wa = a->isFixed() ? 0.0 : 1.0 / a->mass();
            double wb = b->isFixed() ? 0.0 : 1.0 / b->mass();
            double wTotal = wa + wb;
            if (wTotal < 1e-12)
                continue;

            QVector2D correction = (err / static_cast<float>(wTotal)) * dir;
            if (!a->isFixed())
                a->setPos(a->pos() + static_cast<float>(wa) * correction);
            if (!b->isFixed())
                b->setPos(b->pos() - static_cast<float>(wb) * correction);
        }

        if (maxErr < 1e-8)
            break;
    }

    // Correct velocities: remove relative velocity component along each rod
    for (const Rod &rod : m_rods) {
        Bead *a = m_beads[rod.beadA];
        Bead *b = m_beads[rod.beadB];

        QVector2D delta = b->pos() - a->pos();
        float dist = delta.length();
        if (dist < 1e-7f)
            continue;

        QVector2D dir = delta / dist;
        QVector2D relVel = b->vel() - a->vel();
        float vAlong = QVector2D::dotProduct(relVel, dir);

        double wa = a->isFixed() ? 0.0 : 1.0 / a->mass();
        double wb = b->isFixed() ? 0.0 : 1.0 / b->mass();
        double wTotal = wa + wb;
        if (wTotal < 1e-12)
            continue;

        QVector2D vCorr = (vAlong / static_cast<float>(wTotal)) * dir;
        if (!a->isFixed())
            a->setVel(a->vel() + static_cast<float>(wa) * vCorr);
        if (!b->isFixed())
            b->setVel(b->vel() - static_cast<float>(wb) * vCorr);
    }
}

void ChainSimulator::recordFrame(double time)
{
    if (!m_recording)
        return;

    FrameRecord frame;
    frame.time = time;
    frame.positions.reserve(m_beads.size());
    frame.velocities.reserve(m_beads.size());

    for (const Bead *b : m_beads) {
        frame.positions.append(b->pos());
        frame.velocities.append(b->vel());
    }

    m_frames.append(std::move(frame));
}

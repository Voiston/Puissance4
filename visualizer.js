class SoundFX {
    constructor() {
        this.enabled = true;
        this.ctx = null;
    }

    initContext() {
        if (!this.ctx && (window.AudioContext || window.webkitAudioContext)) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        }
    }

    playTone(freq, duration = 0.08, type = 'sine', gainVal = 0.08) {
        if (!this.enabled) return;
        this.initContext();
        if (!this.ctx) return;

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
        gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);

        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start();
        osc.stop(this.ctx.currentTime + duration);
    }

    drop(isHuman = true) { this.playTone(isHuman ? 260 : 340, 0.07, 'triangle', 0.1); }
    reward() { this.playTone(587, 0.1, 'sine', 0.08); setTimeout(() => this.playTone(880, 0.15, 'sine', 0.08), 70); }
    punish() { this.playTone(180, 0.22, 'sawtooth', 0.07); }
    win() { [523, 659, 783, 1046].forEach((f, i) => setTimeout(() => this.playTone(f, 0.14, 'triangle', 0.1), i * 80)); }
}

class NeuralVisualizer {
    constructor(nnCanvasId, chartCanvasId) {
        this.nnCanvas = document.getElementById(nnCanvasId);
        this.nnCtx = this.nnCanvas ? this.nnCanvas.getContext('2d') : null;

        this.chartCanvas = document.getElementById(chartCanvasId);
        this.chartCtx = this.chartCanvas ? this.chartCanvas.getContext('2d') : null;

        this.rewardHistory = [];
    }

    recordGameReward(avgReward) {
        this.rewardHistory.push(avgReward);
        if (this.rewardHistory.length > 80) this.rewardHistory.shift();
        this.drawLearningCurve();
    }

    drawNeuralNetwork(inspection) {
        if (!this.nnCtx || !this.nnCanvas) return;
        const ctx = this.nnCtx;
        const W = this.nnCanvas.width;
        const H = this.nnCanvas.height;

        ctx.clearRect(0, 0, W, H);

        if (!inspection) {
            ctx.fillStyle = '#9aa4b8';
            ctx.font = '14px Poppins';
            ctx.textAlign = 'center';
            ctx.fillText('Jouez un coup ou lancez un entraînement pour observer les activations en direct.', W / 2, H / 2);
            return;
        }

        const layersX = [85, 255, 420, 555, 690];
        const titles = ['Canaux [6×7×2]', 'Attention Conv2D', 'Dense (256)', 'Dense (128)', 'Sortie (7 Col.)'];

        ctx.font = '600 12px Poppins';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#a4b0be';
        layersX.forEach((x, idx) => ctx.fillText(titles[idx], x, 22));

        const getLayerNodes = (x, count, topPad = 48, bottomPad = 32) => {
            const availH = H - topPad - bottomPad;
            const step = availH / Math.max(1, count - 1);
            return Array.from({ length: count }, (_, i) => ({ x, y: topPad + i * step }));
        };

        const convNodes = getLayerNodes(layersX[1], inspection.convSample.length);
        const d256Nodes = getLayerNodes(layersX[2], inspection.dense256Sample.length);
        const d128Nodes = getLayerNodes(layersX[3], inspection.dense128Sample.length);
        const outNodes = getLayerNodes(layersX[4], 7, 62, 48);

        // Synapses
        const drawSynapses = (nodesA, valsA, nodesB, valsB) => {
            for (let i = 0; i < nodesA.length; i++) {
                for (let j = 0; j < nodesB.length; j++) {
                    const strength = Math.min(1, (Math.abs(valsA[i]) + Math.abs(valsB[j])) / 3.5);
                    if (strength < 0.08) continue;
                    ctx.beginPath();
                    ctx.moveTo(nodesA[i].x, nodesA[i].y);
                    ctx.lineTo(nodesB[j].x, nodesB[j].y);
                    ctx.strokeStyle = valsB[j] >= 0
                        ? `rgba(46, 213, 115, ${strength * 0.35})`
                        : `rgba(255, 71, 87, ${strength * 0.35})`;
                    ctx.lineWidth = 0.5 + strength * 1.4;
                    ctx.stroke();
                }
            }
        };

        drawSynapses(convNodes, inspection.convSample, d256Nodes, inspection.dense256Sample);
        drawSynapses(d256Nodes, inspection.dense256Sample, d128Nodes, inspection.dense128Sample);
        drawSynapses(d128Nodes, inspection.dense128Sample, outNodes, inspection.outputs);

        // Mini-grilles Canal 0, Canal 1 et Carte d'Attention Conv2D
        const drawMiniChannel = (centerX, centerY, channelOffset, label, activeColor) => {
            const cellSize = 12.5;
            const gridStartX = centerX - (7 * cellSize) / 2;
            const gridStartY = centerY - (6 * cellSize) / 2;

            ctx.font = '600 10px Poppins';
            ctx.fillStyle = '#9aa4b8';
            ctx.textAlign = 'center';
            ctx.fillText(label, centerX, gridStartY - 7);

            for (let r = 0; r < 6; r++) {
                for (let c = 0; c < 7; c++) {
                    const idx = (r * 7 + c) * 2 + channelOffset;
                    const isActive = inspection.inputs[idx] === 1;
                    const cx = gridStartX + c * cellSize + cellSize / 2;
                    const cy = gridStartY + r * cellSize + cellSize / 2;

                    ctx.beginPath();
                    ctx.arc(cx, cy, 4.2, 0, Math.PI * 2);
                    ctx.fillStyle = isActive ? activeColor : '#23253c';
                    ctx.fill();
                }
            }
        };

        drawMiniChannel(layersX[0], H * 0.28, 0, 'Canal 0 : Moi', '#2ed573');
        drawMiniChannel(layersX[0], H * 0.56, 1, 'Canal 1 : Ennemi', '#ff4757');

        // Carte d'attention spatiale 6x7 en bas à gauche
        if (inspection.attentionMap) {
            const maxAtt = Math.max(0.01, ...inspection.attentionMap);
            const cellSize = 12.5;
            const startX = layersX[0] - (7 * cellSize) / 2;
            const startY = H * 0.84 - (6 * cellSize) / 2;

            ctx.font = '600 10px Poppins';
            ctx.fillStyle = '#ffa502';
            ctx.fillText('🔥 Regard Conv2D', layersX[0], startY - 7);

            for (let r = 0; r < 6; r++) {
                for (let c = 0; c < 7; c++) {
                    const ratio = inspection.attentionMap[r * 7 + c] / maxAtt;
                    const cx = startX + c * cellSize + cellSize / 2;
                    const cy = startY + r * cellSize + cellSize / 2;
                    ctx.beginPath();
                    ctx.arc(cx, cy, 4.2, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(255, 165, 2, ${0.15 + ratio * 0.85})`;
                    ctx.fill();
                }
            }
        }

        // Neurones cachés
        const drawNodes = (nodes, vals, radius = 7) => {
            nodes.forEach((n, i) => {
                const val = vals[i] || 0;
                const intensity = Math.min(1, Math.abs(val) / 2);
                ctx.beginPath();
                ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = val > 0.02
                    ? `rgba(46, 213, 115, ${0.25 + intensity * 0.75})`
                    : '#2b2d48';
                ctx.fill();
                ctx.lineWidth = 1.5;
                ctx.strokeStyle = val > 0.02 ? '#2ed573' : 'rgba(255,255,255,0.15)';
                ctx.stroke();
            });
        };

        drawNodes(convNodes, inspection.convSample, 7);
        drawNodes(d256Nodes, inspection.dense256Sample, 6.5);
        drawNodes(d128Nodes, inspection.dense128Sample, 7.5);

        // Sorties (7 Colonnes)
        outNodes.forEach((n, c) => {
            const isChosen = c === inspection.chosenCol;
            const score = inspection.maskedScores[c];
            ctx.beginPath();
            ctx.arc(n.x, n.y, isChosen ? 11.5 : 8.5, 0, Math.PI * 2);
            ctx.fillStyle = isChosen ? '#2ed573' : (score < 0 ? '#ff4757' : '#5352ed');
            ctx.fill();
            ctx.lineWidth = isChosen ? 2.5 : 1.5;
            ctx.strokeStyle = isChosen ? '#ffffff' : 'rgba(255,255,255,0.25)';
            ctx.stroke();

            ctx.font = isChosen ? '800 11px Poppins' : '600 10px Poppins';
            ctx.fillStyle = isChosen ? '#2ed573' : '#f1f2f6';
            ctx.textAlign = 'left';
            const label = Number.isFinite(score) ? `C${c + 1}: ${score.toFixed(1)}` : `C${c + 1}: Plein`;
            ctx.fillText(label, n.x + 15, n.y + 4);
        });
    }

    drawLearningCurve() {
        if (!this.chartCtx || !this.chartCanvas) return;
        const ctx = this.chartCtx;
        const W = this.chartCanvas.width;
        const H = this.chartCanvas.height;

        ctx.clearRect(0, 0, W, H);

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
        ctx.lineWidth = 1;
        for (let y = 30; y < H; y += 45) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
        }

        if (this.rewardHistory.length < 2) {
            ctx.fillStyle = '#9aa4b8';
            ctx.font = '13px Poppins';
            ctx.textAlign = 'center';
            ctx.fillText('Lancez quelques parties pour tracer la courbe de progression.', W / 2, H / 2);
            return;
        }

        const minVal = Math.min(-12, ...this.rewardHistory);
        const maxVal = Math.max(15, ...this.rewardHistory);
        const range = Math.max(1, maxVal - minVal);

        const zeroY = H - 25 - ((0 - minVal) / range) * (H - 50);
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(25, zeroY);
        ctx.lineTo(W - 25, zeroY);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        this.rewardHistory.forEach((val, idx) => {
            const x = 25 + (idx / (this.rewardHistory.length - 1)) * (W - 50);
            const y = H - 25 - ((val - minVal) / range) * (H - 50);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.strokeStyle = '#2ed573';
        ctx.lineWidth = 2.5;
        ctx.stroke();
    }
}
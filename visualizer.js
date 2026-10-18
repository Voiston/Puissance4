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

    // Notes douces façon marimba / boîte à musique (ondes sinusoïdales rondes)
    playTone(freq, duration = 0.11, type = 'sine', gainVal = 0.06) {
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

    drop(isHuman = true) { this.playTone(isHuman ? 392 : 440, 0.09, 'sine', 0.07); }
    reward() { this.playTone(587, 0.12, 'sine', 0.06); setTimeout(() => this.playTone(784, 0.16, 'sine', 0.06), 80); }
    punish() { this.playTone(293, 0.18, 'triangle', 0.06); setTimeout(() => this.playTone(261, 0.22, 'triangle', 0.06), 90); }
    win() { [523, 659, 784, 1046].forEach((f, i) => setTimeout(() => this.playTone(f, 0.18, 'sine', 0.07), i * 90)); }
    levelUp() {
        [523.25, 659.25, 783.99, 1046.5, 1318.5].forEach((f, i) =>
            setTimeout(() => this.playTone(f, 0.22, 'triangle', 0.08), i * 85)
        );
    }
    save() {
        this.playTone(880, 0.08, 'sine', 0.06);
        setTimeout(() => this.playTone(1174.66, 0.14, 'sine', 0.06), 70);
    }
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
            ctx.fillStyle = '#8a8194';
            ctx.font = '600 14px Fredoka, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Pose un pion ou lance une partie pour voir les petites bulles de neurones s’éveiller !', W / 2, H / 2);
            return;
        }

        const layersX = [85, 255, 420, 555, 685];
        const titles = ['Plateau [6×7×2]', 'Filtres 3×3', 'Idées (256)', 'Tri (128)', 'Choix (7 Col.)'];

        ctx.font = '600 12px Fredoka, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#6d6478';
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

        // Synapses pastel douces
        const drawSynapses = (nodesA, valsA, nodesB, valsB) => {
            for (let i = 0; i < nodesA.length; i++) {
                for (let j = 0; j < nodesB.length; j++) {
                    const strength = Math.min(1, (Math.abs(valsA[i]) + Math.abs(valsB[j])) / 3.5);
                    if (strength < 0.08) continue;
                    ctx.beginPath();
                    ctx.moveTo(nodesA[i].x, nodesA[i].y);
                    ctx.lineTo(nodesB[j].x, nodesB[j].y);
                    ctx.strokeStyle = valsB[j] >= 0
                        ? `rgba(82, 183, 136, ${strength * 0.42})`
                        : `rgba(255, 133, 161, ${strength * 0.42})`;
                    ctx.lineWidth = 0.7 + strength * 1.5;
                    ctx.stroke();
                }
            }
        };

        drawSynapses(convNodes, inspection.convSample, d256Nodes, inspection.dense256Sample);
        drawSynapses(d256Nodes, inspection.dense256Sample, d128Nodes, inspection.dense128Sample);
        drawSynapses(d128Nodes, inspection.dense128Sample, outNodes, inspection.outputs);

        // Mini-grilles Canal 0, Canal 1
        const drawMiniChannel = (centerX, centerY, channelOffset, label, activeColor) => {
            const cellSize = 12.5;
            const gridStartX = centerX - (7 * cellSize) / 2;
            const gridStartY = centerY - (6 * cellSize) / 2;

            ctx.font = '600 10px Fredoka, sans-serif';
            ctx.fillStyle = '#7a7085';
            ctx.textAlign = 'center';
            ctx.fillText(label, centerX, gridStartY - 7);

            for (let r = 0; r < 6; r++) {
                for (let c = 0; c < 7; c++) {
                    const idx = (r * 7 + c) * 2 + channelOffset;
                    const isActive = inspection.inputs[idx] === 1;
                    const cx = gridStartX + c * cellSize + cellSize / 2;
                    const cy = gridStartY + r * cellSize + cellSize / 2;

                    ctx.beginPath();
                    ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
                    ctx.fillStyle = isActive ? activeColor : '#e9e3d5';
                    ctx.fill();
                }
            }
        };

        drawMiniChannel(layersX[0], H * 0.28, 0, 'Canal 0 : Mochi', '#ffca5c');
        drawMiniChannel(layersX[0], H * 0.56, 1, 'Canal 1 : Ami', '#ff85a1');

        // Carte d'attention spatiale
        if (inspection.attentionMap) {
            const maxAtt = Math.max(0.01, ...inspection.attentionMap);
            const cellSize = 12.5;
            const startX = layersX[0] - (7 * cellSize) / 2;
            const startY = H * 0.84 - (6 * cellSize) / 2;

            ctx.font = '600 10px Fredoka, sans-serif';
            ctx.fillStyle = '#d48c00';
            ctx.fillText('🔍 Regard de Mochi', layersX[0], startY - 7);

            for (let r = 0; r < 6; r++) {
                for (let c = 0; c < 7; c++) {
                    const ratio = inspection.attentionMap[r * 7 + c] / maxAtt;
                    const cx = startX + c * cellSize + cellSize / 2;
                    const cy = startY + r * cellSize + cellSize / 2;
                    ctx.beginPath();
                    ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(244, 162, 97, ${0.18 + ratio * 0.82})`;
                    ctx.fill();
                }
            }
        }

        // Bulles des neurones cachés
        const drawNodes = (nodes, vals, radius = 7.5) => {
            nodes.forEach((n, i) => {
                const val = vals[i] || 0;
                const intensity = Math.min(1, Math.abs(val) / 2);
                ctx.beginPath();
                ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = val > 0.02
                    ? `rgba(107, 207, 151, ${0.3 + intensity * 0.7})`
                    : '#f0ebe1';
                ctx.fill();
                ctx.lineWidth = 2;
                ctx.strokeStyle = val > 0.02 ? '#4aa874' : '#dcd4c4';
                ctx.stroke();
            });
        };

        drawNodes(convNodes, inspection.convSample, 7.5);
        drawNodes(d256Nodes, inspection.dense256Sample, 7);
        drawNodes(d128Nodes, inspection.dense128Sample, 8);

        // Sorties (7 Colonnes)
        outNodes.forEach((n, c) => {
            const isChosen = c === inspection.chosenCol;
            const score = inspection.maskedScores[c];
            ctx.beginPath();
            ctx.arc(n.x, n.y, isChosen ? 12 : 9, 0, Math.PI * 2);
            ctx.fillStyle = isChosen ? '#6bcf97' : (score < 0 ? '#ff85a1' : '#a3c4f3');
            ctx.fill();
            ctx.lineWidth = isChosen ? 3 : 2;
            ctx.strokeStyle = isChosen ? '#4aa874' : '#ffffff';
            ctx.stroke();

            ctx.font = isChosen ? '700 11px Fredoka, sans-serif' : '600 10px Fredoka, sans-serif';
            ctx.fillStyle = isChosen ? '#2d7a50' : '#4a4453';
            ctx.textAlign = 'left';
            const label = Number.isFinite(score) ? `C${c + 1}: ${score.toFixed(1)}` : `C${c + 1}: Plein`;
            ctx.fillText(label, n.x + 16, n.y + 4);
        });
    }

    drawLearningCurve() {
        if (!this.chartCtx || !this.chartCanvas) return;
        const ctx = this.chartCtx;
        const W = this.chartCanvas.width;
        const H = this.chartCanvas.height;

        ctx.clearRect(0, 0, W, H);

        ctx.strokeStyle = '#ebe4d6';
        ctx.lineWidth = 1;
        for (let y = 30; y < H; y += 45) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
        }

        if (this.rewardHistory.length < 2) {
            ctx.fillStyle = '#8a8194';
            ctx.font = '600 13px Fredoka, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Joue ou entraîne Mochi pour dessiner sa courbe de progrès !', W / 2, H / 2);
            return;
        }

        const minVal = Math.min(-12, ...this.rewardHistory);
        const maxVal = Math.max(15, ...this.rewardHistory);
        const range = Math.max(1, maxVal - minVal);

        const zeroY = H - 25 - ((0 - minVal) / range) * (H - 50);
        ctx.beginPath();
        ctx.setLineDash([5, 5]);
        ctx.moveTo(25, zeroY);
        ctx.lineTo(W - 25, zeroY);
        ctx.strokeStyle = '#c8bfb0';
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        this.rewardHistory.forEach((val, idx) => {
            const x = 25 + (idx / (this.rewardHistory.length - 1)) * (W - 50);
            const y = H - 25 - ((val - minVal) / range) * (H - 50);
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.strokeStyle = '#52b788';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.stroke();
    }
}
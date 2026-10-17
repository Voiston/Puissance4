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

        const layersX = [90, 255, 415, 555, 690];
        const titles = ['Entrée [6×7×2]', 'Conv2D (3×3)', 'Dense (256)', 'Dense (128)', 'Sortie (7 Col.)'];

        // En-têtes des couches
        ctx.font = '600 12px Poppins';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#a4b0be';
        layersX.forEach((x, idx) => ctx.fillText(titles[idx], x, 22));

        // Coordonnées verticales des neurones
        const getLayerNodes = (x, count, topPad = 48, bottomPad = 32) => {
            const availH = H - topPad - bottomPad;
            const step = availH / Math.max(1, count - 1);
            return Array.from({ length: count }, (_, i) => ({ x, y: topPad + i * step }));
        };

        const convNodes = getLayerNodes(layersX[1], inspection.convSample.length);
        const d256Nodes = getLayerNodes(layersX[2], inspection.dense256Sample.length);
        const d128Nodes = getLayerNodes(layersX[3], inspection.dense128Sample.length);
        const outNodes = getLayerNodes(layersX[4], 7, 62, 48);

        // 1. Dessin des synapses (connexions pondérées par l'activation)
        const drawSynapses = (nodesA, valsA, nodesB, valsB) => {
            for (let i = 0; i < nodesA.length; i++) {
                for (let j = 0; j < nodesB.length; j++) {
                    const strength = Math.min(1, (Math.abs(valsA[i]) + Math.abs(valsB[j])) / 3.5);
                    if (strength < 0.08) continue;
                    ctx.beginPath();
                    ctx.moveTo(nodesA[i].x, nodesA[i].y);
                    ctx.lineTo(nodesB[j].x, nodesB[j].y);
                    const isPositive = valsB[j] >= 0;
                    ctx.strokeStyle = isPositive
                        ? `rgba(46, 213, 115, ${strength * 0.38})`
                        : `rgba(255, 71, 87, ${strength * 0.38})`;
                    ctx.lineWidth = 0.5 + strength * 1.5;
                    ctx.stroke();
                }
            }
        };

        drawSynapses(convNodes, inspection.convSample, d256Nodes, inspection.dense256Sample);
        drawSynapses(d256Nodes, inspection.dense256Sample, d128Nodes, inspection.dense128Sample);
        drawSynapses(d128Nodes, inspection.dense128Sample, outNodes, inspection.outputs);

        // 2. Dessin des 2 canaux d'entrée [6, 7, 2] (Canal 0 : Allié, Canal 1 : Ennemi)
        const drawMiniChannel = (centerY, channelOffset, label, activeColor) => {
            const cellSize = 13;
            const gridStartX = layersX[0] - (7 * cellSize) / 2;
            const gridStartY = centerY - (6 * cellSize) / 2;

            ctx.font = '600 10px Poppins';
            ctx.fillStyle = '#9aa4b8';
            ctx.textAlign = 'center';
            ctx.fillText(label, layersX[0], gridStartY - 8);

            for (let r = 0; r < 6; r++) {
                for (let c = 0; c < 7; c++) {
                    const idx = (r * 7 + c) * 2 + channelOffset;
                    const isActive = inspection.inputs[idx] === 1;
                    const cx = gridStartX + c * cellSize + cellSize / 2;
                    const cy = gridStartY + r * cellSize + cellSize / 2;

                    if (isActive) {
                        const targetNode = convNodes[(r * 7 + c) % convNodes.length];
                        ctx.beginPath();
                        ctx.moveTo(cx, cy);
                        ctx.lineTo(targetNode.x, targetNode.y);
                        ctx.strokeStyle = channelOffset === 0
                            ? 'rgba(46, 213, 115, 0.22)'
                            : 'rgba(255, 71, 87, 0.22)';
                        ctx.lineWidth = 1;
                        ctx.stroke();
                    }

                    ctx.beginPath();
                    ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
                    ctx.fillStyle = isActive ? activeColor : '#23253c';
                    ctx.fill();
                }
            }
        };

        drawMiniChannel(H * 0.32, 0, 'Canal 0 : Mes Pions', '#2ed573');
        drawMiniChannel(H * 0.72, 1, 'Canal 1 : Adversaire', '#ff4757');

        // 3. Dessin des neurones cachés (Conv2D, Dense 256, Dense 128)
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

        // 4. Dessin des 7 neurones de sortie (Q-Values par colonne)
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

        // Lignes de repère horizontales
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

        // Ligne zéro de référence
        const zeroY = H - 25 - ((0 - minVal) / range) * (H - 50);
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(25, zeroY);
        ctx.lineTo(W - 25, zeroY);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.stroke();
        ctx.setLineDash([]);

        // Courbe des récompenses
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
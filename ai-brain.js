class NeuralAgentManager {
    constructor(rows = 6, cols = 7) {
        this.ROWS = rows;
        this.COLS = cols;
        this.GAMMA = 0.99;
        this.MEMORY_SIZE = 50000;

        this.agents = {
            'A': { model: null, target: null, probe: null, memory: [], games: 0, storage: 'indexeddb://dqn-ia-a-v5', ageKey: 'dqn_age_a_v5' },
            'B': { model: null, target: null, probe: null, memory: [], games: 0, storage: 'indexeddb://dqn-ia-b-v5', ageKey: 'dqn_age_b_v5' }
        };

        this.lastInspection = null;
    }

    compile(model) {
        model.compile({ optimizer: tf.train.adam(0.00025), loss: tf.losses.huberLoss });
        return model;
    }

    // Réseau convolutif avec entrée bi-canal [6, 7, 2] (84 valeurs)
    createSequentialNetwork() {
        const m = tf.sequential();
        m.add(tf.layers.reshape({ targetShape: [6, 7, 2], inputShape: [84] }));
        m.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'same' }));
        m.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu', padding: 'same' }));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        m.add(tf.layers.dense({ units: 7, activation: 'linear' }));
        return this.compile(m);
    }

    buildProbe(seqModel) {
        return tf.model({
            inputs: seqModel.inputs,
            outputs: [
                seqModel.layers[2].output,
                seqModel.layers[4].output,
                seqModel.layers[5].output,
                seqModel.layers[6].output
            ]
        });
    }

    async init() {
        for (const key of ['A', 'B']) {
            const ag = this.agents[key];
            ag.games = parseInt(localStorage.getItem(ag.ageKey) || '0', 10);
            try {
                ag.model = await tf.loadLayersModel(ag.storage);
                ag.target = await tf.loadLayersModel(ag.storage);
                this.compile(ag.model);
                this.compile(ag.target);
            } catch (e) {
                ag.model = this.createSequentialNetwork();
                ag.target = this.createSequentialNetwork();
                ag.games = 0;
            }
            ag.probe = this.buildProbe(ag.model);
        }
    }

    async wipeAll() {
        for (const key of ['A', 'B']) {
            const ag = this.agents[key];
            try { await tf.io.removeModel(ag.storage); } catch (e) {}
            if (ag.model) ag.model.dispose();
            if (ag.target) ag.target.dispose();

            ag.model = this.createSequentialNetwork();
            ag.target = this.createSequentialNetwork();
            ag.probe = this.buildProbe(ag.model);
            ag.memory = [];
            ag.games = 0;
            localStorage.setItem(ag.ageKey, '0');
        }
    }

    // Encodage bi-canal [6, 7, 2] : Canal 0 (Moi), Canal 1 (Adversaire)
    getNormalizedState(grid, perspectivePlayer) {
        const encoded = new Float32Array(84);
        const flat = grid.flat();
        for (let i = 0; i < 42; i++) {
            if (flat[i] === perspectivePlayer) {
                encoded[i * 2] = 1;
            } else if (flat[i] !== 0) {
                encoded[i * 2 + 1] = 1;
            }
        }
        return Array.from(encoded);
    }

    getMirroredState(flatBoard) {
        const mirrored = new Array(84).fill(0);
        for (let r = 0; r < 6; r++) {
            for (let c = 0; c < 7; c++) {
                const srcIdx = (r * 7 + c) * 2;
                const dstCol = 6 - c;
                const dstIdx = (r * 7 + dstCol) * 2;
                mirrored[dstIdx] = flatBoard[srcIdx];
                mirrored[dstIdx + 1] = flatBoard[srcIdx + 1];
            }
        }
        return mirrored;
    }

    selectAction(grid, aiKey, perspectivePlayer, epsilon = 0, captureActivations = true) {
        const validCols = [0, 1, 2, 3, 4, 5, 6].filter(c => grid[0][c] === 0);
        if (validCols.length === 0) return { col: undefined, scores: [], exploratory: false };

        const normState = this.getNormalizedState(grid, perspectivePlayer);

        const { rawScores, inspection } = tf.tidy(() => {
            const input = tf.tensor2d([normState]);
            const ag = this.agents[aiKey];

            if (!captureActivations) {
                const out = ag.model.predict(input);
                return { rawScores: Array.from(out.dataSync()), inspection: null };
            }

            const [convOut, d256Out, d128Out, qOut] = ag.probe.predict(input);
            const qArr = Array.from(qOut.dataSync());

            const sample = (arr, count) => {
                const res = [];
                const step = Math.max(1, Math.floor(arr.length / count));
                for (let i = 0; i < count; i++) res.push(arr[(i * step) % arr.length]);
                return res;
            };

            return {
                rawScores: qArr,
                inspection: {
                    aiKey,
                    inputs: normState,
                    convSample: sample(convOut.dataSync(), 12),
                    dense256Sample: sample(d256Out.dataSync(), 14),
                    dense128Sample: sample(d128Out.dataSync(), 10),
                    outputs: qArr
                }
            };
        });

        const maskedScores = rawScores.map((s, c) => grid[0][c] === 0 ? s : -Infinity);

        let chosenCol = validCols[0];
        let exploratory = false;

        if (Math.random() < epsilon) {
            chosenCol = validCols[Math.floor(Math.random() * validCols.length)];
            exploratory = true;
        } else {
            for (const c of validCols) {
                if (maskedScores[c] > maskedScores[chosenCol]) chosenCol = c;
            }
        }

        if (inspection) {
            inspection.chosenCol = chosenCol;
            inspection.maskedScores = maskedScores;
            this.lastInspection = inspection;
        }

        return { col: chosenCol, scores: maskedScores, exploratory, normState };
    }

    saveMemory(aiKey, normState, action, reward, normNextState, done, clones = 1) {
        const memory = this.agents[aiKey].memory;
        const mirroredState = this.getMirroredState(normState);
        const mirroredNextState = this.getMirroredState(normNextState);
        const mirroredAction = (this.COLS - 1) - action;

        for (let i = 0; i < clones; i++) {
            memory.push({ state: normState, action, reward, nextState: normNextState, done });
            memory.push({ state: mirroredState, action: mirroredAction, reward, nextState: mirroredNextState, done });
        }
        while (memory.length > this.MEMORY_SIZE) memory.shift();
    }

    // Entraînement avec Double DQN & Équation de Bellman Negamax
    async trainBatch(aiKey, batchSize = 128) {
        const ag = this.agents[aiKey];
        if (ag.memory.length < batchSize) return;

        const batch = [];
        for (let i = 0; i < batchSize; i++) {
            batch.push(ag.memory[Math.floor(Math.random() * ag.memory.length)]);
        }

        const { x, y } = tf.tidy(() => {
            const states = tf.tensor2d(batch.map(m => m.state));
            const nextStates = tf.tensor2d(batch.map(m => m.nextState));

            const currentQ = ag.model.predict(states);
            const nextQMain = ag.model.predict(nextStates);
            const nextQTarget = ag.target.predict(nextStates);

            const qValues = currentQ.arraySync();
            const nextQMainValues = nextQMain.arraySync();
            const nextQTargetValues = nextQTarget.arraySync();

            batch.forEach((m, i) => {
                let target = m.reward;
                if (!m.done) {
                    // Sélection du meilleur coup dans l'état suivant
                    let bestFutureAction = 0;
                    let bestVal = -Infinity;
                    for (let c = 0; c < 7; c++) {
                        if (nextQMainValues[i][c] > bestVal) {
                            bestVal = nextQMainValues[i][c];
                            bestFutureAction = c;
                        }
                    }
                    const futureValue = nextQTargetValues[i][bestFutureAction];
                    // Negamax : ce qui est bon pour l'adversaire au tour suivant est soustrait
                    target = m.reward - this.GAMMA * futureValue;
                }
                qValues[i][m.action] = target;
            });

            return { x: states, y: tf.tensor2d(qValues) };
        });

        await ag.model.fit(x, y, { epochs: 1, silent: true });
        x.dispose();
        y.dispose();
    }

    async checkpoint(aiKey) {
        const ag = this.agents[aiKey];
        ag.target.setWeights(ag.model.getWeights());
        await ag.model.save(ag.storage);
        localStorage.setItem(ag.ageKey, ag.games.toString());
    }
}
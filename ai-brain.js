class NeuralAgentManager {
    constructor(rows = 6, cols = 7) {
        this.ROWS = rows;
        this.COLS = cols;
        this.GAMMA = 0.99;
        this.MEMORY_SIZE = 50000;
        this.SLOTS_META_KEY = 'mochi_save_slots_v1';

        this.RANKS = [
            { min: 0,    title: "🐣 Œuf Curieux",     next: 30 },
            { min: 30,   title: "👶 Apprenti Mochi",  next: 150 },
            { min: 150,  title: "🛡️ Gardien Pastel",  next: 500 },
            { min: 500,  title: "🧠 Stratège Éveillé", next: 1200 },
            { min: 1200, title: "👑 Grand Maître",    next: 2500 },
            { min: 2500, title: "✨ Divinité Mochi",  next: Infinity }
        ];

        this.agents = {
            'A': {
                name: 'Mochi-A',
                model: null, target: null, probe: null, memory: [],
                games: 0, blocks: 0, wins: 0,
                storage: 'indexeddb://dqn-ia-a-v5',
                metaKey: 'dqn_meta_a_v5'
            },
            'B': {
                name: 'Mochi-B',
                model: null, target: null, probe: null, memory: [],
                games: 0, blocks: 0, wins: 0,
                storage: 'indexeddb://dqn-ia-b-v5',
                metaKey: 'dqn_meta_b_v5'
            }
        };

        this.lastInspection = null;
    }

    compile(model) {
        model.compile({ optimizer: tf.train.adam(0.00025), loss: tf.losses.huberLoss });
        return model;
    }

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
            const savedMeta = JSON.parse(localStorage.getItem(ag.metaKey) || 'null');
            const legacyAge = parseInt(localStorage.getItem(`dqn_age_${key.toLowerCase()}_v5`) || '0', 10);

            ag.games = savedMeta ? (savedMeta.games || 0) : legacyAge;
            ag.blocks = savedMeta ? (savedMeta.blocks || 0) : 0;
            ag.wins = savedMeta ? (savedMeta.wins || 0) : 0;
            ag.name = savedMeta && savedMeta.name ? savedMeta.name : `Mochi-${key}`;

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

    // --- PROGRESSION RPG (NIVEAU, RANG & BARRE D'XP) ---
    getProgression(games) {
        const level = Math.max(1, Math.floor(Math.sqrt(games / 2)) + 1);
        let currentRank = this.RANKS[0];
        for (const r of this.RANKS) {
            if (games >= r.min) currentRank = r;
        }
        const span = currentRank.next === Infinity ? 1000 : (currentRank.next - currentRank.min);
        const progressInRank = games - currentRank.min;
        const pct = currentRank.next === Infinity
            ? 100
            : Math.min(100, Math.max(4, Math.round((progressInRank / span) * 100)));

        return {
            level,
            rankTitle: currentRank.title,
            nextGoal: currentRank.next,
            xpPercent: pct
        };
    }

    getRankTitle(games) {
        return this.getProgression(games).rankTitle;
    }

    // --- SYSTÈME DE CARTOUCHES (4 SLOTS INTERNES) ---
    getSaveSlots() {
        const raw = localStorage.getItem(this.SLOTS_META_KEY);
        if (!raw) return { 1: null, 2: null, 3: null, 4: null };
        try {
            return JSON.parse(raw);
        } catch {
            return { 1: null, 2: null, 3: null, 4: null };
        }
    }

    async saveToSlot(slotId, aiKey = 'A', customName = '') {
        const ag = this.agents[aiKey];
        const slotStorage = `indexeddb://mochi-cartridge-slot-${slotId}`;
        await ag.model.save(slotStorage);

        const slots = this.getSaveSlots();
        const prog = this.getProgression(ag.games);
        slots[slotId] = {
            slotId,
            name: customName.trim() || ag.name || `Mochi-${aiKey}`,
            games: ag.games,
            blocks: ag.blocks,
            wins: ag.wins,
            level: prog.level,
            rankTitle: prog.rankTitle,
            date: new Date().toLocaleDateString('fr-FR', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
        };
        localStorage.setItem(this.SLOTS_META_KEY, JSON.stringify(slots));
        return slots[slotId];
    }

    async loadFromSlot(slotId, targetKey = 'A') {
        const slots = this.getSaveSlots();
        const meta = slots[slotId];
        if (!meta) throw new Error("Emplacement vide");

        const slotStorage = `indexeddb://mochi-cartridge-slot-${slotId}`;
        const loadedModel = await tf.loadLayersModel(slotStorage);
        const loadedTarget = await tf.loadLayersModel(slotStorage);

        const dst = this.agents[targetKey];
        if (dst.model) dst.model.dispose();
        if (dst.target) dst.target.dispose();

        dst.model = this.compile(loadedModel);
        dst.target = this.compile(loadedTarget);
        dst.probe = this.buildProbe(dst.model);
        dst.name = meta.name;
        dst.games = meta.games || 0;
        dst.blocks = meta.blocks || 0;
        dst.wins = meta.wins || 0;
        dst.memory = [];

        await this.checkpoint(targetKey);
        return meta;
    }

    async deleteSlot(slotId) {
        const slotStorage = `indexeddb://mochi-cartridge-slot-${slotId}`;
        try { await tf.io.removeModel(slotStorage); } catch (e) {}
        const slots = this.getSaveSlots();
        slots[slotId] = null;
        localStorage.setItem(this.SLOTS_META_KEY, JSON.stringify(slots));
    }

    // --- EXPORT / IMPORT EN 1 SEUL FICHIER (.mochi) ---
    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        const chunk = 0x8000;
        for (let i = 0; i < bytes.length; i += chunk) {
            binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
        }
        return btoa(binary);
    }

    base64ToFloat32Array(base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return new Float32Array(bytes.buffer);
    }

    async exportSingleFile(aiKey = 'A') {
        const ag = this.agents[aiKey];
        const weights = ag.model.getWeights();
        const serializedWeights = weights.map(w => {
            const f32 = new Float32Array(w.dataSync());
            return {
                shape: w.shape,
                b64: this.arrayBufferToBase64(f32.buffer)
            };
        });

        const payload = {
            format: 'MOCHI_CONNECT4_V1',
            name: ag.name || `Mochi-${aiKey}`,
            games: ag.games,
            blocks: ag.blocks,
            wins: ag.wins,
            exportedAt: new Date().toISOString(),
            weights: serializedWeights
        };

        const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const safeName = (ag.name || `mochi-${aiKey}`).toLowerCase().replace(/[^a-z0-9-_]/g, '-');
        a.href = url;
        a.download = `${safeName}-niv${this.getProgression(ag.games).level}.mochi`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }

    async importSingleFile(file, targetKey = 'A') {
        const text = await file.text();
        const data = JSON.parse(text);
        if (!data || data.format !== 'MOCHI_CONNECT4_V1' || !Array.isArray(data.weights)) {
            throw new Error("Fichier .mochi invalide");
        }

        const tensors = data.weights.map(w => {
            const vals = this.base64ToFloat32Array(w.b64);
            return tf.tensor(vals, w.shape, 'float32');
        });

        const dst = this.agents[targetKey];
        dst.model.setWeights(tensors);
        dst.target.setWeights(tensors);
        tensors.forEach(t => t.dispose());

        dst.name = data.name || `Mochi-${targetKey}`;
        dst.games = data.games || 0;
        dst.blocks = data.blocks || 0;
        dst.wins = data.wins || 0;
        dst.memory = [];

        await this.checkpoint(targetKey);
        return dst;
    }

    async cloneAgent(fromKey = 'A', toKey = 'B') {
        const src = this.agents[fromKey];
        const dst = this.agents[toKey];
        dst.model.setWeights(src.model.getWeights());
        dst.target.setWeights(src.target.getWeights());
        dst.games = src.games;
        dst.blocks = src.blocks;
        dst.wins = src.wins;
        dst.name = `${src.name} (Clone)`;
        await this.checkpoint(toKey);
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
            ag.blocks = 0;
            ag.wins = 0;
            ag.name = `Mochi-${key}`;
            localStorage.removeItem(ag.metaKey);
            localStorage.setItem(`dqn_age_${key.toLowerCase()}_v5`, '0');
        }
    }

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
            const convArr = convOut.dataSync();
            const qArr = Array.from(qOut.dataSync());

            const attentionMap = new Array(42).fill(0);
            for (let cellIdx = 0; cellIdx < 42; cellIdx++) {
                let sum = 0;
                const base = cellIdx * 128;
                for (let f = 0; f < 128; f++) sum += convArr[base + f];
                attentionMap[cellIdx] = sum / 128;
            }

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
                    attentionMap,
                    convSample: sample(convArr, 12),
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
                    let bestFutureAction = 0;
                    let bestVal = -Infinity;
                    for (let c = 0; c < 7; c++) {
                        const colFull = m.nextState[c * 2] !== 0 || m.nextState[c * 2 + 1] !== 0;
                        if (!colFull && nextQMainValues[i][c] > bestVal) {
                            bestVal = nextQMainValues[i][c];
                            bestFutureAction = c;
                        }
                    }
                    const futureValue = Number.isFinite(bestVal) ? nextQTargetValues[i][bestFutureAction] : 0;
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
        localStorage.setItem(ag.metaKey, JSON.stringify({
            name: ag.name,
            games: ag.games,
            blocks: ag.blocks,
            wins: ag.wins
        }));
        localStorage.setItem(`dqn_age_${aiKey.toLowerCase()}_v5`, ag.games.toString());
    }
}
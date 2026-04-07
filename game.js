const ROWS = 6, COLS = 7;
let board = [];
let model, targetModel;
let isTraining = false;

// Paramètres DQN Optimisés
const GAMMA = 0.97;
const MEMORY_SIZE = 5000; // Taille de mémoire stable pour éviter la saturation RAM
let memory = []; 

let totalGames = parseInt(localStorage.getItem('dqn_pure_totalGames')) || 0;
let aiWins = parseInt(localStorage.getItem('dqn_pure_aiWins')) || 0;

// 1. INITIALISATION DU PLATEAU
function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

function renderBoard() {
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    const fragment = document.createDocumentFragment();
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell' + (board[r][c] === 1 ? ' player' : board[r][c] === 2 ? ' ai' : '');
            div.onclick = () => handleMove(c);
            fragment.appendChild(div);
        }
    }
    gridEl.appendChild(fragment);
}

// 2. ARCHITECTURE DU RÉSEAU (Mode Pur sans règles codées)
async function initIA() {
    const createModel = () => {
        const m = tf.sequential();
        m.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        m.add(tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({units: 128, activation: 'relu'}));
        m.add(tf.layers.dense({units: 7, activation: 'linear'}));
        m.compile({optimizer: tf.train.adam(0.00025), loss: 'meanSquaredError'});
        return m;
    };

    try {
        // Chargement de la version stable v2
        model = await tf.loadLayersModel('localstorage://dqn-pure-v2');
        targetModel = await tf.loadLayersModel('localstorage://dqn-pure-v2');
        document.getElementById('ia-status').innerText = "DQN Pur : Chargé";
    } catch (e) {
        model = createModel();
        targetModel = createModel();
        document.getElementById('ia-status').innerText = "DQN Pur : Initialisé";
    }
    updateStatsDisplay();
    renderBoard();
}

// 3. LOGIQUE DE PRÉDICTION SÉCURISÉE
function getBestMove(grid, epsilon = 0) {
    if (Math.random() < epsilon) return Math.floor(Math.random() * COLS);
    
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        const pred = model.predict(input);
        return pred.argMax(1).dataSync()[0];
    });
}

// 4. ENTRAÎNEMENT AVEC GESTION MÉMOIRE RIGOUREUSE
async function trainBatch(size = 64) {
    if (memory.length < size) return;

    const loss = tf.tidy(() => {
        const batch = [];
        for(let i=0; i<size; i++) batch.push(memory[Math.floor(Math.random() * memory.length)]);

        const states = tf.tensor2d(batch.map(m => m.state));
        const nextStates = tf.tensor2d(batch.map(m => m.nextState));
        
        const currentQ = model.predict(states);
        const nextQ = targetModel.predict(nextStates);

        const qValues = currentQ.arraySync();
        const nextQValues = nextQ.arraySync();

        batch.forEach((m, i) => {
            let target = m.reward;
            if (!m.done) target = m.reward + GAMMA * Math.max(...nextQValues[i]);
            qValues[i][m.action] = target;
        });

        return { states, qValues: tf.tensor2d(qValues) };
    });

    await model.fit(loss.states, loss.qValues, {epochs: 1, silent: true});
    
    loss.states.dispose();
    loss.qValues.dispose();
}

// 5. SIMULATION RAFALE ANTI-CRASH (Visualisation 1 match sur 10)
async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 250; 

    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let turn = (Math.random() < 0.5) ? 1 : 2;
        let epsilon = Math.max(0.05, 0.4 - (i / batchSize));
        
        // On ne rend visuel que certains matchs pour soulager le processeur
        let showVisual = (i % 10 === 0);

        while (true) {
            let state = [...board.flat()];
            let col = getBestMove(board, epsilon);
            
            if (board[0][col] !== 0) {
                col = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
                if (col === undefined) break; 
            }

            if (dropToken(board, col, turn)) {
                let isWin = checkWinner(board, turn);
                let done = isWin || board.flat().every(v => v !== 0);
                
                // Récompenses DQN
                let reward = 0.02; // Bonus de survie
                if (isWin) reward = (turn === 2 ? 15 : -15);

                memory.push({state, action: col, reward, nextState: [...board.flat()], done});
                if (memory.length > MEMORY_SIZE) memory.shift();

                if (showVisual) {
                    renderBoard();
                    await new Promise(requestAnimationFrame); 
                }

                if (done) { if(isWin && turn === 2) aiWins++; break; }
                turn = (turn === 1) ? 2 : 1;
            } else break;
        }

        totalGames++;
        await trainBatch(64);

        if (i % 10 === 0) {
            targetModel.setWeights(model.getWeights());
            updateStatsDisplay();
            document.getElementById('status').innerText = `Simulation : ${i}/${batchSize}`;
            
            // PAUSE CRUCIAL : laisse le navigateur vider la RAM toutes les 10 parties
            await new Promise(resolve => setTimeout(resolve, 50)); 
        }
    }

    await model.save('localstorage://dqn-pure-v2');
    isTraining = false;
    document.getElementById('status').innerText = "Entraînement terminé avec succès !";
    initBoard(); renderBoard();
}

// 6. FONCTIONS TECHNIQUES (Règles du jeu)
function dropToken(g, c, p) {
    if (c < 0 || c >= COLS || g[0][c] !== 0) return false;
    for (let r = ROWS - 1; r >= 0; r--) { if (g[r][c] === 0) { g[r][c] = p; return true; } }
    return false;
}

function checkWinner(g, p) {
    for (let r=0; r<6; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r][c+1]===p && g[r][c+2]===p && g[r][c+3]===p) return true;
    for (let r=0; r<3; r++) for (let c=0; c<7; c++) if (g[r][c]===p && g[r+1][c]===p && g[r+2][c]===p && g[r+3][c]===p) return true;
    for (let r=3; r<6; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r-1][c+1]===p && g[r-2][c+2]===p && g[r-3][c+3]===p) return true;
    for (let r=0; r<3; r++) for (let c=0; c<4; c++) if (g[r][c]===p && g[r+1][c+1]===p && g[r+2][c+2]===p && g[r+3][c+3]===p) return true;
    return false;
}

function updateStatsDisplay() {
    document.getElementById('total-games').innerText = totalGames;
    const rate = totalGames > 0 ? ((aiWins / totalGames) * 100).toFixed(1) : 0;
    document.getElementById('ai-winrate').innerText = rate;
    localStorage.setItem('dqn_pure_totalGames', totalGames);
    localStorage.setItem('dqn_pure_aiWins', aiWins);
}

async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;
    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) { totalGames++; updateStatsDisplay(); document.getElementById('status').innerText = "Victoire Humaine !"; return; }
        
        setTimeout(() => {
            let aiCol = getBestMove(board);
            if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];
            dropToken(board, aiCol, 2);
            renderBoard();
            if (checkWinner(board, 2)) { totalGames++; aiWins++; updateStatsDisplay(); document.getElementById('status').innerText = "L'IA a gagné !"; }
        }, 200);
    }
}

// Boutons
document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); document.getElementById('status').innerText = "Prêt."; };
document.getElementById('btn-train').onclick = runTraining;
document.getElementById('btn-save').onclick = () => model.save('localstorage://dqn-pure-v2');

// Lancement initial
initBoard();
initIA();

const ROWS = 6, COLS = 7;
let board = [];
let isTraining = false;
let isArena = false;

// Paramètres DQN
const GAMMA = 0.96;
const MEMORY_SIZE = 2000;

// Nos deux gladiateurs
const AIs = {
    'A': { model: null, target: null, memory: [], storage: 'localstorage://dqn-ia-a' },
    'B': { model: null, target: null, memory: [], storage: 'localstorage://dqn-ia-b' }
};

// 1. INITIALISATION DU PLATEAU ET RENDU VISUEL
function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

function renderBoard() {
    // Si on s'entraîne, on ne dessine RIEN pour économiser 100% du GPU visuel
    if (isTraining) return; 

    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    const fragment = document.createDocumentFragment();
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell' + (board[r][c] === 1 ? ' player' : board[r][c] === 2 ? ' ai' : '');
            // En mode arène, on bloque les clics humains
            div.onclick = () => { if(!isArena) handleMove(c); };
            fragment.appendChild(div);
        }
    }
    gridEl.appendChild(fragment);
}

// 2. CRÉATION DES DEUX CERVEAUX
async function initIA() {
    const createModel = () => {
        const m = tf.sequential();
        m.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        m.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({units: 64, activation: 'relu'}));
        m.add(tf.layers.dense({units: 7, activation: 'linear'}));
        m.compile({optimizer: tf.train.adam(0.0005), loss: 'meanSquaredError'});
        return m;
    };

    // Charger ou créer IA-A
    try { AIs['A'].model = await tf.loadLayersModel(AIs['A'].storage); AIs['A'].target = await tf.loadLayersModel(AIs['A'].storage); }
    catch (e) { AIs['A'].model = createModel(); AIs['A'].target = createModel(); }

    // Charger ou créer IA-B
    try { AIs['B'].model = await tf.loadLayersModel(AIs['B'].storage); AIs['B'].target = await tf.loadLayersModel(AIs['B'].storage); }
    catch (e) { AIs['B'].model = createModel(); AIs['B'].target = createModel(); }

    document.getElementById('ia-status').innerText = "IA A et B prêtes";
    renderBoard();
}

// 3. PRÉDICTION SÉCURISÉE
function getBestMove(grid, aiName, epsilon = 0) {
    if (Math.random() < epsilon) {
        const validCols = [0,1,2,3,4,5,6].filter(c => grid[0][c] === 0);
        return validCols[Math.floor(Math.random() * validCols.length)];
    }

    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        const pred = AIs[aiName].model.predict(input);
        return pred.argMax(1).dataSync()[0];
    });
}

// 4. ENTRAÎNEMENT D'UNE IA SPÉCIFIQUE
async function trainBatch(aiName, size = 32) {
    const memory = AIs[aiName].memory;
    if (memory.length < size) return;

    tf.tidy(() => {
        const batch = [];
        for(let i=0; i<size; i++) batch.push(memory[Math.floor(Math.random() * memory.length)]);
        
        const states = tf.tensor2d(batch.map(m => m.state));
        const nextStates = tf.tensor2d(batch.map(m => m.nextState));
        
        const currentQ = AIs[aiName].model.predict(states);
        const nextQ = AIs[aiName].target.predict(nextStates);
        
        const qValues = currentQ.arraySync();
        const nextQValues = nextQ.arraySync();

        batch.forEach((m, i) => {
            let target = m.reward;
            if (!m.done) target = m.reward + GAMMA * Math.max(...nextQValues[i]);
            qValues[i][m.action] = target;
        });

        AIs[aiName].model.fit(states, tf.tensor2d(qValues), {epochs: 1, silent: true});
    });
}

// 5. BOUCLE D'ENTRAÎNEMENT INVISIBLE (Le mode Béton)
async function runTraining(aiName) {
    if (isTraining || isArena) return;
    isTraining = true;
    const batchSize = 150; 
    const statusText = document.getElementById('status');

    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let turn = (Math.random() < 0.5) ? 1 : 2; // Elle joue les deux rôles contre elle-même
        let epsilon = Math.max(0.1, 0.4 - (i / batchSize));

        while (true) {
            let state = [...board.flat()];
            let col = getBestMove(board, aiName, epsilon);
            
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
            if (col === undefined) break;

            if (dropToken(board, col, turn)) {
                let win = checkWinner(board, turn);
                let done = win || board.flat().every(v => v !== 0);
                
                let reward = win ? 20 : 0.05; // On ne punit pas l'autre rôle car c'est la même IA
                if(done && !win) reward = 2; // Match nul = un peu positif

                AIs[aiName].memory.push({state, action: col, reward, nextState: [...board.flat()], done});
                if (AIs[aiName].memory.length > MEMORY_SIZE) AIs[aiName].memory.shift();
                
                if (done) break;
                turn = (turn === 1) ? 2 : 1;
            } else break;
        }

        await trainBatch(aiName, 32);

        // MISE À JOUR DU TEXTE UNIQUEMENT (Pas de rendu de grille)
        statusText.innerText = `Entraînement IA-${aiName} : ${i} / ${batchSize}`;

        if (i % 10 === 0) {
            AIs[aiName].target.setWeights(AIs[aiName].model.getWeights());
            // Pause vitale pour laisser la RAM se purger
            await new Promise(r => setTimeout(r, 50));
        }
    }

    await AIs[aiName].model.save(AIs[aiName].storage);
    isTraining = false;
    statusText.innerText = `IA-${aiName} optimisée et sauvegardée !`;
    initBoard(); renderBoard();
}

// 6. LE COMBAT DES TITANS (IA-A vs IA-B)
async function runArena() {
    if (isTraining || isArena) return;
    isArena = true;
    initBoard(); renderBoard();
    
    document.getElementById('status').innerText = "⚔️ COMBAT : IA-A (Rouge) vs IA-B (Jaune) ⚔️";
    await new Promise(r => setTimeout(r, 1000));

    let turn = 1; // IA-A commence toujours (Rouge)
    
    while (true) {
        let activeAI = turn === 1 ? 'A' : 'B';
        let col = getBestMove(board, activeAI, 0); // Epsilon 0 = Sérieux absolu
        
        if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
        if (col === undefined) {
            document.getElementById('status').innerText = "Match Nul entre les deux IA !";
            break;
        }

        if (dropToken(board, col, turn)) {
            renderBoard();
            await new Promise(r => setTimeout(r, 250)); // On ralentit pour que tu puisses regarder

            if (checkWinner(board, turn)) {
                document.getElementById('status').innerText = `🏆 L'IA-${activeAI} a terrassé son adversaire !`;
                break;
            }
            if (board.flat().every(v => v !== 0)) {
                 document.getElementById('status').innerText = "Égalité parfaite !";
                 break;
            }
            turn = (turn === 1) ? 2 : 1;
        } else break;
    }
    isArena = false;
}

// 7. JOUER MANUELLEMENT CONTRE IA-A
async function handleMove(col) {
    if (isTraining || isArena || board[0][col] !== 0) return;

    if (dropToken(board, col, 1)) { // Humain = Rouge
        renderBoard();
        if (checkWinner(board, 1)) { document.getElementById('status').innerText = "Tu as battu l'IA-A !"; return; }

        document.getElementById('status').innerText = "L'IA-A réfléchit...";
        await new Promise(r => setTimeout(r, 150));

        let aiCol = getBestMove(board, 'A', 0); // On joue contre l'IA A
        if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);

        if (aiCol !== undefined && dropToken(board, aiCol, 2)) { // IA = Jaune
            renderBoard();
            if (checkWinner(board, 2)) document.getElementById('status').innerText = "L'IA-A t'a écrasé !";
            else if (board.flat().every(v => v !== 0)) document.getElementById('status').innerText = "Nul !";
            else document.getElementById('status').innerText = "À toi.";
        }
    }
}

// FONCTIONS STANDARDS
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

// BINDING DES BOUTONS
document.getElementById('btn-reset').onclick = () => { isArena = false; isTraining = false; initBoard(); renderBoard(); document.getElementById('status').innerText = "À toi vs IA-A."; };
document.getElementById('btn-train-a').onclick = () => runTraining('A');
document.getElementById('btn-train-b').onclick = () => runTraining('B');
document.getElementById('btn-arena').onclick = () => runArena();

// Lancement au chargement
initBoard(); 
initIA();

const ROWS = 6;
const COLS = 7;
let board = [];
let model;
let isTraining = false;

// 1. ARCHITECTURE "VISION SPATIALE" (CNN)
async function initIA() {
    try {
        // Tentative de chargement du modèle PRO
        model = await tf.loadLayersModel('localstorage://c4-ultra-pro');
        document.getElementById('status').innerText = "IA ULTRA-PRO Chargée";
    } catch (e) {
        // Création d'un cerveau capable de détecter les lignes (Filtres 4x4)
        model = tf.sequential();
        model.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        model.add(tf.layers.conv2d({filters: 64, kernelSize: 4, activation: 'relu', padding: 'same'}));
        model.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'}));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({units: 128, activation: 'relu'}));
        model.add(tf.layers.dense({units: 7, activation: 'linear'}));
        
        model.compile({optimizer: tf.train.adam(0.0005), loss: 'meanSquaredError'});
        document.getElementById('status').innerText = "Nouveau Cerveau CNN Initialisé.";
    }
    renderBoard();
}

// 2. RENDU ULTRA-FLUIDE
function renderBoard() {
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    const fragment = document.createDocumentFragment();
    
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell';
            if (board[r][c] === 1) div.classList.add('player');
            if (board[r][c] === 2) div.classList.add('ai');
            div.onclick = () => handleMove(c);
            fragment.appendChild(div);
        }
    }
    gridEl.appendChild(fragment);
}

// 3. LOGIQUE DE JEU OPTIMISÉE
function dropToken(grid, col, player) {
    if (col < 0 || col >= COLS || grid[0][col] !== 0) return false;
    for (let r = ROWS - 1; r >= 0; r--) {
        if (grid[r][col] === 0) {
            grid[r][col] = player;
            return true;
        }
    }
    return false;
}

function checkWinner(g, p) {
    // Horizontal
    for (let r=0; r<6; r++) for (let c=0; c<4; c++) 
        if (g[r][c]===p && g[r][c+1]===p && g[r][c+2]===p && g[r][c+3]===p) return true;
    // Vertical
    for (let r=0; r<3; r++) for (let c=0; c<7; c++) 
        if (g[r][c]===p && g[r+1][c]===p && g[r+2][c]===p && g[r+3][c]===p) return true;
    // Diagonales
    for (let r=3; r<6; r++) for (let c=0; c<4; c++) 
        if (g[r][c]===p && g[r-1][c+1]===p && g[r-2][c+2]===p && g[r-3][c+3]===p) return true;
    for (let r=0; r<3; r++) for (let c=0; c<4; c++) 
        if (g[r][c]===p && g[r+1][c+1]===p && g[r+2][c+2]===p && g[r+3][c+3]===p) return true;
    return false;
}

// 4. PRÉDICTION (Intelligence)
function getBestMove(grid) {
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        const pred = model.predict(input);
        return pred.argMax(1).dataSync()[0];
    });
}

// 5. ENTRAÎNEMENT RAFALE (100 MATCHS VISIBLES)
async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 100;
    let allStates = [], allLabels = [];

    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let moves = [];
        let turn = (i % 2 === 0) ? 1 : 2;
        let winner = 0;

        document.getElementById('status').innerText = `Rafale : Match ${i}/${batchSize}`;

        for (let step = 0; step < 42; step++) {
            // IA joue avec 15% d'exploration (hasard)
            let col = Math.random() < 0.15 ? Math.floor(Math.random() * 7) : getBestMove(board);
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];

            if (dropToken(board, col, turn)) {
                moves.push({state: board.flat(), move: col, player: turn});
                
                // Mise à jour visuelle synchronisée sur l'écran (Ultra-Fast)
                renderBoard();
                await new Promise(requestAnimationFrame); 
                
                if (checkWinner(board, turn)) { winner = turn; break; }
                turn = (turn === 1) ? 2 : 1;
            }
        }

        // Attribution des récompenses (Q-Learning)
        moves.forEach(m => {
            let label = new Array(7).fill(0);
            if (winner === 2) label[m.move] = (m.player === 2) ? 1.0 : -1.0;
            if (winner === 1) label[m.move] = (m.player === 2) ? -1.0 : 0.2;
            allStates.push(m.state);
            allLabels.push(label);
        });

        // Apprentissage par petits blocs pour éviter les fuites mémoire
        if (i % 10 === 0) {
            const xs = tf.tensor2d(allStates);
            const ys = tf.tensor2d(allLabels);
            await model.fit(xs, ys, {epochs: 1, shuffle: true});
            xs.dispose(); ys.dispose();
            allStates = []; allLabels = [];
        }
    }

    isTraining = false;
    document.getElementById('status').innerText = "Entraînement terminé !";
    initBoard();
    renderBoard();
}

// 6. ACTION JOUEUR HUMAIN
async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;

    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) { document.getElementById('status').innerText = "Gagné !"; return; }

        document.getElementById('status').innerText = "L'IA analyse...";
        setTimeout(() => {
            let aiCol = getBestMove(board);
            if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];
            
            if (dropToken(board, aiCol, 2)) {
                renderBoard();
                if (checkWinner(board, 2)) document.getElementById('status').innerText = "L'IA a gagné !";
                else document.getElementById('status').innerText = "À vous de jouer.";
            }
        }, 300);
    }
}

// BOUTONS
document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); document.getElementById('status').innerText = "Prêt."; };
document.getElementById('btn-train').onclick = runTraining;
document.getElementById('btn-save').onclick = async () => { 
    await model.save('localstorage://c4-ultra-pro'); 
    alert("Mémoire ULTRA-PRO sauvegardée !"); 
};

// LANCEMENT
function initBoard() { board = Array(ROWS).fill().map(() => Array(COLS).fill(0)); }
initBoard();
initIA();

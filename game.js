const ROWS = 6;
const COLS = 7;
let board = [];
let model;
let isTraining = false;

// 1. INITIALISATION DU CERVEAU (Architecture Convolutionnelle Pro)
async function initIA() {
    try {
        model = await tf.loadLayersModel('localstorage://c4-ultra-hardcore');
        document.getElementById('status').innerText = "IA Hardcore Chargée";
    } catch (e) {
        model = tf.sequential();
        // Le Reshape permet à l'IA de voir la grille comme une image 6x7
        model.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        model.add(tf.layers.conv2d({filters: 64, kernelSize: 4, activation: 'relu', padding: 'same'}));
        model.add(tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu', padding: 'same'}));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({units: 128, activation: 'relu'}));
        model.add(tf.layers.dense({units: 7, activation: 'linear'}));
        
        model.compile({optimizer: tf.train.adam(0.0005), loss: 'meanSquaredError'});
        document.getElementById('status').innerText = "Nouveau Cerveau CNN (Anticipation) créé.";
    }
    renderBoard();
}

// 2. LOGIQUE DE JEU & VÉRIFICATION
function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

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

// 3. PRÉDICTION AVEC ANTICIPATION (Bloque l'adversaire avant de réfléchir)
function getBestMove(grid) {
    // ÉTAPE 1 : Est-ce que je peux gagner ce tour-ci ?
    for (let c = 0; c < COLS; c++) {
        let tempBoard = grid.map(row => [...row]);
        if (dropToken(tempBoard, c, 2)) {
            if (checkWinner(tempBoard, 2)) return c;
        }
    }
    // ÉTAPE 2 : Est-ce que l'humain peut gagner au prochain tour ? SI OUI, JE BLOQUE !
    for (let c = 0; c < COLS; c++) {
        let tempBoard = grid.map(row => [...row]);
        if (dropToken(tempBoard, c, 1)) {
            if (checkWinner(tempBoard, 1)) return c;
        }
    }
    // ÉTAPE 3 : Sinon, utiliser le réseau de neurones pour la stratégie
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        const pred = model.predict(input);
        return pred.argMax(1).dataSync()[0];
    });
}

// 4. RENDU VISUEL
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

// 5. ENTRAÎNEMENT RAFALE (100 MATCHS)
async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const batchSize = 100;
    let allStates = [], allLabels = [];

    for (let i = 1; i <= batchSize; i++) {
        initBoard();
        let moves = [];
        let turn = (Math.random() < 0.5) ? 1 : 2;
        let winner = 0;

        document.getElementById('status').innerText = `Simulation Rafale : Match ${i}/${batchSize}`;

        for (let step = 0; step < 42; step++) {
            let col = Math.random() < 0.2 ? Math.floor(Math.random() * 7) : getBestMove(board);
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];

            if (dropToken(board, col, turn)) {
                moves.push({state: board.flat(), move: col, player: turn});
                renderBoard();
                await new Promise(requestAnimationFrame); 
                
                if (checkWinner(board, turn)) { winner = turn; break; }
                turn = (turn === 1) ? 2 : 1;
            }
        }

        // Apprentissage par renforcement (Punition Sévère)
        moves.forEach(m => {
            let label = new Array(7).fill(0);
            if (winner === 2 && m.player === 2) label[m.move] = 1.0; 
            if (winner === 1 && m.player === 2) label[m.move] = -2.0; // Grosse punition en cas de défaite
            allStates.push(m.state);
            allLabels.push(label);
        });

        if (i % 10 === 0) {
            const xs = tf.tensor2d(allStates);
            const ys = tf.tensor2d(allLabels);
            await model.fit(xs, ys, {epochs: 2, shuffle: true});
            xs.dispose(); ys.dispose();
            allStates = []; allLabels = [];
        }
    }

    isTraining = false;
    document.getElementById('status').innerText = "Batch terminé. Mémoire mise à jour !";
    initBoard();
    renderBoard();
}

// 6. TOUR DU JOUEUR HUMAIN
async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;

    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) { document.getElementById('status').innerText = "Vous avez gagné !"; return; }

        document.getElementById('status').innerText = "L'IA anticipe...";
        setTimeout(() => {
            let aiCol = getBestMove(board);
            if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];
            
            if (dropToken(board, aiCol, 2)) {
                renderBoard();
                if (checkWinner(board, 2)) document.getElementById('status').innerText = "L'IA a gagné !";
                else document.getElementById('status').innerText = "À vous de jouer.";
            }
        }, 200);
    }
}

// ÉVÉNEMENTS
document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); document.getElementById('status').innerText = "Prêt."; };
document.getElementById('btn-train').onclick = runTraining;
document.getElementById('btn-save').onclick = async () => { 
    await model.save('localstorage://c4-ultra-hardcore'); 
    alert("Cerveau Hardcore Sauvegardé !"); 
};

initBoard();
initIA();

const ROWS = 6;
const COLS = 7;
let board = [];
let model;
let isTraining = false;

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// 1. Initialisation de la grille
function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

// 2. Création ou chargement du modèle (Cerveau Boosté à 128 neurones)
async function initIA() {
    try {
        model = await tf.loadLayersModel('localstorage://c4-model');
        document.getElementById('status').innerText = "IA Chargée (Cerveau 128 neurones)";
    } catch (e) {
        model = tf.sequential();
        // Couche 1 : 128 neurones pour mieux mémoriser les motifs
        model.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [42]}));
        // Couche 2 : 64 neurones pour la décision
        model.add(tf.layers.dense({units: 64, activation: 'relu'}));
        // Sortie : 7 colonnes
        model.add(tf.layers.dense({units: 7, activation: 'linear'}));
        model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
        document.getElementById('status').innerText = "Nouveau Cerveau 128px créé.";
    }
    renderBoard();
}

// 3. Affichage
function renderBoard() {
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    board.forEach((row, r) => {
        row.forEach((cell, c) => {
            const div = document.createElement('div');
            div.className = 'cell';
            if (cell === 1) div.classList.add('player');
            if (cell === 2) div.classList.add('ai');
            div.onclick = () => handleMove(c);
            gridEl.appendChild(div);
        });
    });
}

// 4. Logique de chute
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

// 5. Action Joueur
async function handleMove(col) {
    if (isTraining || board[0][col] !== 0) return;

    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) {
            document.getElementById('status').innerText = "Vous avez gagné !";
            return;
        }

        document.getElementById('status').innerText = "L'IA réfléchit...";
        setTimeout(() => {
            let aiCol = getBestMove(board);
            // Sécurité anti-colonne pleine
            if (board[0][aiCol] !== 0) {
                const free = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0);
                aiCol = free[Math.floor(Math.random() * free.length)];
            }
            
            if (dropToken(board, aiCol, 2)) {
                renderBoard();
                if (checkWinner(board, 2)) {
                    document.getElementById('status').innerText = "L'IA gagne !";
                } else {
                    document.getElementById('status').innerText = "À vous.";
                }
            }
        }, 400);
    }
}

// 6. Prédiction
function getBestMove(grid) {
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        return model.predict(input).argMax(1).dataSync()[0];
    });
}

// 7. Victoire (Simplifiée mais efficace)
function checkWinner(grid, p) {
    // Horizontal
    for (let r = 0; r < ROWS; r++) 
        for (let c = 0; c < COLS - 3; c++) 
            if (grid[r][c]===p && grid[r][c+1]===p && grid[r][c+2]===p && grid[r][c+3]===p) return true;
    // Vertical
    for (let r = 0; r < ROWS - 3; r++) 
        for (let c = 0; c < COLS; c++) 
            if (grid[r][c]===p && grid[r+1][c]===p && grid[r+2][c]===p && grid[r+3][c]===p) return true;
    // Diagonales (Optionnel pour l'entraînement rapide, mais mieux ici)
    for (let r = 3; r < ROWS; r++)
        for (let c = 0; c < COLS - 3; c++)
            if (grid[r][c]===p && grid[r-1][c+1]===p && grid[r-2][c+2]===p && grid[r-3][c+3]===p) return true;
    for (let r = 0; r < ROWS - 3; r++)
        for (let c = 0; c < COLS - 3; c++)
            if (grid[r][c]===p && grid[r+1][c+1]===p && grid[r+2][c+2]===p && grid[r+3][c+3]===p) return true;
    return false;
}

// 8. Entraînement Visuel avec Apprentissage par Renforcement
async function runTraining() {
    if (isTraining) return;
    isTraining = true;
    const iterations = 15; 

    for (let i = 1; i <= iterations; i++) {
        initBoard();
        let moves = [];
        let turn = 1;
        let winner = 0;

        document.getElementById('status').innerText = `Entraînement : Match ${i}/${iterations}`;

        for (let step = 0; step < 42; step++) {
            // L'IA joue (80% intelligent, 20% hasard pour explorer)
            let col = Math.random() < 0.2 ? Math.floor(Math.random() * 7) : getBestMove(board);
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].filter(c => board[0][c] === 0)[0];

            if (dropToken(board, col, turn)) {
                // On enregistre l'état de la grille AVANT le coup et le coup joué
                moves.push({state: board.flat(), move: col, player: turn});
                renderBoard();
                
                if (checkWinner(board, turn)) {
                    winner = turn;
                    break;
                }
                turn = turn === 1 ? 2 : 1;
                await sleep(40); 
            }
        }

        // --- PHASE D'APPRENTISSAGE ---
        // On prépare les données pour le réseau de neurones
        const states = tf.tensor2d(moves.map(m => m.state));
        const labels = tf.tensor2d(moves.map(m => {
            let l = new Array(7).fill(0);
            // Si le joueur 2 (IA) gagne, on met 1 sur ses coups. Si elle perd, on met -1.
            if (winner === 2 && m.player === 2) l[m.move] = 1; 
            if (winner === 1 && m.player === 2) l[m.move] = -1; 
            return l;
        }));

        await model.fit(states, labels, {epochs: 1});
        await sleep(200);
    }

    isTraining = false;
    document.getElementById('status').innerText = "IA Mise à jour !";
    initBoard();
    renderBoard();
}

// Boutons
document.getElementById('btn-reset').onclick = () => { initBoard(); renderBoard(); document.getElementById('status').innerText = "Nouvelle partie."; };
document.getElementById('btn-train').onclick = runTraining;
document.getElementById('btn-save').onclick = async () => { await model.save('localstorage://c4-model'); alert("Mémoire sauvegardée !"); };

initBoard();
initIA();

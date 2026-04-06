const ROWS = 6;
const COLS = 7;
let board = [];
let model;
let isTraining = false;

// 1. Initialisation de la grille
function initBoard() {
    board = Array(ROWS).fill().map(() => Array(COLS).fill(0));
}

// 2. Création ou chargement du modèle
async function initIA() {
    try {
        model = await tf.loadLayersModel('localstorage://c4-model');
        document.getElementById('status').innerText = "IA chargée ! À vous.";
    } catch (e) {
        model = tf.sequential();
        model.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [42]}));
        model.add(tf.layers.dense({units: 64, activation: 'relu'}));
        model.add(tf.layers.dense({units: 7, activation: 'linear'}));
        model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
        document.getElementById('status').innerText = "Nouvelle IA créée.";
    }
    renderBoard();
}

// 3. Affichage de la grille
function renderBoard() {
    const gridEl = document.getElementById('board');
    gridEl.innerHTML = '';
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            if (board[r][c] === 1) cell.classList.add('player');
            if (board[r][c] === 2) cell.classList.add('ai');
            cell.onclick = () => handleMove(c);
            gridEl.appendChild(cell);
        }
    }
}

// 4. Logique pour faire tomber un pion
function dropToken(grid, col, player) {
    for (let r = ROWS - 1; r >= 0; r--) {
        if (grid[r][col] === 0) {
            grid[r][col] = player;
            return true;
        }
    }
    return false; // Colonne pleine
}

// 5. Action du joueur
async function handleMove(col) {
    if (isTraining) return;
    
    if (dropToken(board, col, 1)) {
        renderBoard();
        if (checkWinner(board, 1)) {
            document.getElementById('status').innerText = "Gagné ! Bravo.";
            return;
        }

        document.getElementById('status').innerText = "L'IA réfléchit...";
        setTimeout(() => {
            const aiCol = getBestMove(board);
            if (dropToken(board, aiCol, 2)) {
                renderBoard();
                if (checkWinner(board, 2)) {
                    document.getElementById('status').innerText = "L'IA a gagné !";
                } else {
                    document.getElementById('status').innerText = "À vous de jouer.";
                }
            }
        }, 500);
    }
}

// 6. L'IA choisit le meilleur coup
function getBestMove(grid) {
    return tf.tidy(() => {
        const input = tf.tensor2d([grid.flat()]);
        const prediction = model.predict(input);
        return prediction.argMax(1).dataSync()[0];
    });
}

// 7. Vérification de victoire
function checkWinner(grid, p) {
    // Horizontal
    for (let r = 0; r < ROWS; r++) 
        for (let c = 0; c < COLS - 3; c++) 
            if (grid[r][c]===p && grid[r][c+1]===p && grid[r][c+2]===p && grid[r][c+3]===p) return true;
    // Vertical
    for (let r = 0; r < ROWS - 3; r++) 
        for (let c = 0; c < COLS; c++) 
            if (grid[r][c]===p && grid[r+1][c]===p && grid[r+2][c]===p && grid[r+3][c]===p) return true;
    return false;
}

// 8. Entraînement automatique (Self-Play)
async function runTraining() {
    isTraining = true;
    const iterations = 20;
    for (let i = 1; i <= iterations; i++) {
        document.getElementById('status').innerText = `Entraînement : ${i}/${iterations}`;
        let localBoard = Array(ROWS).fill().map(() => Array(COLS).fill(0));
        let moves = [];
        let turn = 1;
        
        for (let step = 0; step < 42; step++) {
            let col = Math.random() < 0.3 ? Math.floor(Math.random() * 7) : getBestMove(localBoard);
            if (dropToken(localBoard, col, turn)) {
                moves.push({state: localBoard.flat(), move: col, player: turn});
                if (checkWinner(localBoard, turn)) break;
                turn = turn === 1 ? 2 : 1;
            }
        }
        // Apprentissage simple : on renforce les coups du dernier joueur (gagnant potentiel)
        const states = tf.tensor2d(moves.map(m => m.state));
        const labels = tf.tensor2d(moves.map(m => {
            let l = Array(7).fill(0);
            l[m.move] = 1;
            return l;
        }));
        await model.fit(states, labels, {epochs: 1});
        await tf.nextFrame();
    }
    isTraining = false;
    initBoard();
    renderBoard();
    document.getElementById('status').innerText = "Entraînement fini !";
}

// Événements des boutons
document.getElementById('btn-reset').onclick = () => {
    initBoard();
    renderBoard();
    document.getElementById('status').innerText = "Nouvelle partie.";
};

document.getElementById('btn-train').onclick = runTraining;

document.getElementById('btn-save').onclick = async () => {
    await model.save('localstorage://c4-model');
    alert("IA Sauvegardée !");
};

// Lancement
initBoard();
initIA();

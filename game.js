const ROWS = 6, COLS = 7;
let board = [];
let isTraining = false;
let isArena = false;
let stopTrainingRequested = false; // Nouveau : permet d'arrêter la boucle

// Paramètres DQN "PC Edition"
const GAMMA = 0.99; // Vision très long terme
const MEMORY_SIZE = 80000; // Utilisation de la RAM du PC pour un meilleur historique

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
    if (isTraining) return; // Économie totale du GPU pour le calcul pur
    const gridEl = document.getElementById('board');
    if (!gridEl) return;
    gridEl.innerHTML = '';
    const fragment = document.createDocumentFragment();
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const div = document.createElement('div');
            div.className = 'cell' + (board[r][c] === 1 ? ' player' : board[r][c] === 2 ? ' ai' : '');
            div.onclick = () => { if(!isArena) handleMove(c); };
            fragment.appendChild(div);
        }
    }
    gridEl.appendChild(fragment);
}

// 2. CRÉATION DES DEUX CERVEAUX
async function initIA() {
    // On extrait la compilation dans une fonction à part pour pouvoir la réutiliser
    const compileModel = (m) => {
        m.compile({optimizer: tf.train.adam(0.00025), loss: 'meanSquaredError'});
        return m;
    };

    const createModel = () => {
        const m = tf.sequential();
        m.add(tf.layers.reshape({targetShape: [6, 7, 1], inputShape: [42]}));
        m.add(tf.layers.conv2d({filters: 128, kernelSize: 4, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.conv2d({filters: 128, kernelSize: 4, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.conv2d({filters: 128, kernelSize: 3, activation: 'relu', padding: 'same'}));
        m.add(tf.layers.flatten());
        m.add(tf.layers.dense({units: 512, activation: 'relu'}));
        m.add(tf.layers.dense({units: 256, activation: 'relu'}));
        m.add(tf.layers.dense({units: 7, activation: 'linear'}));
        
        return compileModel(m); // On compile le nouveau modèle
    };

    // Charger ou créer IA-A
    try { 
        AIs['A'].model = await tf.loadLayersModel(AIs['A'].storage); 
        AIs['A'].target = await tf.loadLayersModel(AIs['A'].storage); 
        
        compileModel(AIs['A'].model);
        compileModel(AIs['A'].target);
    }
    catch (e) { 
        AIs['A'].model = createModel(); 
        AIs['A'].target = createModel(); 
    }

    // Charger ou créer IA-B
    try { 
        AIs['B'].model = await tf.loadLayersModel(AIs['B'].storage); 
        AIs['B'].target = await tf.loadLayersModel(AIs['B'].storage); 
        
        compileModel(AIs['B'].model);
        compileModel(AIs['B'].target);
    }
    catch (e) { 
        AIs['B'].model = createModel(); 
        AIs['B'].target = createModel(); 
    }

    document.getElementById('ia-status').innerText = "IA A et B prêtes (Mode Haute Performance)";
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
async function trainBatch(aiName, size = 512) {
    const memory = AIs[aiName].memory;
    if (memory.length < size) return;

    // 1. On prépare notre lot d'exemples aléatoires
    const batch = [];
    for(let i=0; i<size; i++) batch.push(memory[Math.floor(Math.random() * memory.length)]);

    // 2. On utilise tf.tidy pour calculer nos valeurs sans fuite de mémoire.
    const { x, y } = tf.tidy(() => {
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

        return { x: states, y: tf.tensor2d(qValues) };
    });

    // 3. On attend PROPREMENT la fin de l'entraînement en dehors du tf.tidy
    await AIs[aiName].model.fit(x, y, {epochs: 1, silent: true});

    // 4. On nettoie manuellement x et y maintenant que fit() a terminé
    x.dispose();
    y.dispose();
}

async function runTraining(aiName) {
    if (isTraining || isArena) return;
    isTraining = true;
    stopTrainingRequested = false; // On réinitialise la demande d'arrêt
    
    let gamesPlayed = 0; // On compte les parties au lieu d'avoir un nombre fixe
    const statusText = document.getElementById('status');
    statusText.innerText = `Entraînement IA-${aiName} démarré...`;

    // Boucle infinie, qui s'arrête si on passe stopTrainingRequested à true
    while (!stopTrainingRequested) {
        gamesPlayed++;
        initBoard();
        let turn = (Math.random() < 0.5) ? 1 : 2; 
        
        // Epsilon diminue en continu, mais se bloque à 10% d'exploration minimum
        let epsilon = Math.max(0.10, 0.8 - (gamesPlayed / 10000));

        while (true) {
            let state = [...board.flat()];
            let col = getBestMove(board, aiName, epsilon);
            
            if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
            if (col === undefined) break;

            if (dropToken(board, col, turn)) {
                let win = checkWinner(board, turn);
                let done = win || board.flat().every(v => v !== 0);
                
                let reward = calculateReward(board, col, turn, win, done && !win);

                AIs[aiName].memory.push({state, action: col, reward, nextState: [...board.flat()], done});
                if (AIs[aiName].memory.length > MEMORY_SIZE) AIs[aiName].memory.shift();
                
                if (done) break;
                turn = (turn === 1) ? 2 : 1;
            } else break;
        }

        // On entraîne le réseau
        await trainBatch(aiName, 512); 

        // Mise à jour de l'affichage toutes les 10 parties pour ne pas saturer l'écran
        if (gamesPlayed % 10 === 0) {
            statusText.innerText = `Entraînement IA-${aiName} : ${gamesPlayed} parties jouées (Clique sur ton bouton pour stopper)`;
        }

        // Toutes les 200 parties : stabilité et SAUVEGARDE AUTO
        if (gamesPlayed % 200 === 0) {
            AIs[aiName].target.setWeights(AIs[aiName].model.getWeights());
            await AIs[aiName].model.save(AIs[aiName].storage); // Sauvegarde en arrière-plan !
        }
        
        // CRUCIAL : Laisse le navigateur respirer 1 milliseconde pour éviter le crash
        // et permettre d'écouter le clic sur le bouton Stop.
        await tf.nextFrame(); 
    }

    // -- FIN DE L'ENTRAÎNEMENT (Quand on a cliqué sur Stop) --
    await AIs[aiName].model.save(AIs[aiName].storage);
    isTraining = false;
    statusText.innerText = `Entraînement stoppé ! IA-${aiName} optimisée sur ${gamesPlayed} parties.`;
    initBoard(); renderBoard();
}
// 6. LE COMBAT DES TITANS (IA-A vs IA-B)
async function runArena() {
    if (isTraining || isArena) return;
    isArena = true;
    initBoard(); renderBoard();
    
    document.getElementById('status').innerText = "⚔️ COMBAT : IA-A (Rouge) vs IA-B (Jaune) ⚔️";
    await new Promise(r => setTimeout(r, 1000));

    let turn = 1; 
    
    while (true) {
        let activeAI = turn === 1 ? 'A' : 'B';
        let col = getBestMove(board, activeAI, 0); 
        
        if (board[0][col] !== 0) col = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);
        if (col === undefined) {
            document.getElementById('status').innerText = "Match Nul entre les deux IA !";
            break;
        }

        if (dropToken(board, col, turn)) {
            renderBoard();
            await new Promise(r => setTimeout(r, 300)); 

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

// 7. JOUER MANUELLEMENT CONTRE IA-A (Maintenant avec apprentissage en direct !)
async function handleMove(col) {
    if (isTraining || isArena || board[0][col] !== 0) return;

    // --- TOUR DU JOUEUR (Toi) ---
    if (dropToken(board, col, 1)) { 
        renderBoard();
        let humanWon = checkWinner(board, 1);
        let isDraw = board.flat().every(v => v !== 0);

        if (humanWon || isDraw) { 
            document.getElementById('status').innerText = humanWon ? "Tu as battu l'IA-A !" : "Nul !"; 
            return; 
        }

        document.getElementById('status').innerText = "L'IA-A réfléchit...";
        await new Promise(r => setTimeout(r, 50)); 

        // --- TOUR DE L'IA ---
        let stateBeforeAI = [...board.flat()]; // L'IA mémorise le plateau AVANT de jouer
        
        let aiCol = getBestMove(board, 'A', 0); 
        if (board[0][aiCol] !== 0) aiCol = [0,1,2,3,4,5,6].find(c => board[0][c] === 0);

        if (aiCol !== undefined && dropToken(board, aiCol, 2)) { 
            renderBoard();
            
            let aiWon = checkWinner(board, 2);
            let aiDraw = board.flat().every(v => v !== 0);
            let done = aiWon || aiDraw;

            // 🧠 1. L'IA CALCULE SA RÉCOMPENSE
            let reward = calculateReward(board, aiCol, 2, aiWon, aiDraw);

            // 📝 2. ELLE PREND DES NOTES DANS SA MÉMOIRE
            AIs['A'].memory.push({
                state: stateBeforeAI, 
                action: aiCol, 
                reward: reward, 
                nextState: [...board.flat()], 
                done: done
            });

            // On s'assure que la mémoire ne déborde pas
            if (AIs['A'].memory.length > MEMORY_SIZE) AIs['A'].memory.shift();

            // 🏋️ 3. ELLE S'ENTRAÎNE FURTIVEMENT EN ARRIÈRE-PLAN
            // On lance un petit batch de 128 souvenirs pour ajuster ses neurones immédiatement
            trainBatch('A', 128);

            if (aiWon) document.getElementById('status').innerText = "L'IA-A t'a écrasé !";
            else if (aiDraw) document.getElementById('status').innerText = "Nul !";
            else document.getElementById('status').innerText = "À toi.";
        }
    }
}

// 8. FONCTION DE RÉCOMPENSE AVANCÉE (Reward Shaping)
function calculateReward(board, lastAction, currentPlayer, isWin, isDraw) {
    // 1. Les fins de parties priment sur tout
    if (isWin) return 100;
    if (isDraw) return 2;

    let reward = 0;
    const opponent = (currentPlayer === 1) ? 2 : 1;

    // 2. Micro-récompense stratégique : jouer au centre
    if (lastAction === 3) {
        reward += 2; 
    } else if (lastAction === 2 || lastAction === 4) {
        reward += 1; 
    }

    // 3. Récompenses d'Analyse
    if (didBlockOpponentWin(board, lastAction, opponent)) {
        reward += 50; // Bloquer une défaite imminente
    }
    
    if (createdLineOfThree(board, lastAction, currentPlayer)) {
        reward += 10; // Créer une opportunité
    }

    return reward;
}

// FONCTIONS STANDARDS ET UTILITAIRES D'ANALYSE
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

function didBlockOpponentWin(g, col, opponent) {
    let r = 0;
    while (r < ROWS && g[r][col] === 0) r++;
    if (r === ROWS) return false; 

    const originalColor = g[r][col];
    g[r][col] = opponent; // Test fictif pour l'adversaire
    
    const blocked = checkWinner(g, opponent);
    
    g[r][col] = originalColor; // Restauration
    return blocked;
}

function createdLineOfThree(g, col, player) {
    let r = 0;
    while (r < ROWS && g[r][col] === 0) r++;
    if (r === ROWS) return false;

    let linesOfThree = 0;

    const checkWindow = (r1, c1, r2, c2, r3, c3, r4, c4) => {
        if (r1<0 || r1>=ROWS || c1<0 || c1>=COLS ||
            r4<0 || r4>=ROWS || c4<0 || c4>=COLS) return;

        const window = [g[r1][c1], g[r2][c2], g[r3][c3], g[r4][c4]];
        const playerCount = window.filter(v => v === player).length;
        const emptyCount = window.filter(v => v === 0).length;

        if (playerCount === 3 && emptyCount === 1) linesOfThree++;
    };

    for (let i = 0; i < 4; i++) {
        checkWindow(r, col-3+i, r, col-2+i, r, col-1+i, r, col+i);
        checkWindow(r+i, col, r+1+i, col, r+2+i, col, r+3+i, col);
        checkWindow(r-3+i, col-3+i, r-2+i, col-2+i, r-1+i, col-1+i, r+i, col+i);
        checkWindow(r-3+i, col+3-i, r-2+i, col+2-i, r-1+i, col+1-i, r+i, col-i);
    }

    return linesOfThree > 0;
}

// BINDING DES BOUTONS
document.getElementById('btn-reset').onclick = () => { 
    isArena = false; stopTrainingRequested = true; 
    initBoard(); renderBoard(); 
    document.getElementById('status').innerText = "À toi vs IA-A."; 
};

// Fonction pour gérer le toggle Start/Stop
function toggleTraining(aiName, btnId, originalText) {
    const btn = document.getElementById(btnId);
    if (isTraining && !stopTrainingRequested) {
        // Si ça tourne, on demande l'arrêt
        stopTrainingRequested = true;
        btn.innerText = "Arrêt en cours...";
    } else if (!isTraining) {
        // Si c'est à l'arrêt, on lance
        btn.innerText = `🛑 Stopper l'IA-${aiName}`;
        runTraining(aiName).then(() => {
            btn.innerText = originalText; // On remet le texte normal à la fin
        });
    }
}

document.getElementById('btn-train-a').onclick = () => toggleTraining('A', 'btn-train-a', "Entraîner IA-A");
document.getElementById('btn-train-b').onclick = () => toggleTraining('B', 'btn-train-b', "Entraîner IA-B");
document.getElementById('btn-arena').onclick = () => runArena();

// Lancement au chargement
initBoard(); 
initIA();

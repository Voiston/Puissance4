class Connect4Engine {
    constructor(rows = 6, cols = 7) {
        this.ROWS = rows;
        this.COLS = cols;
        this.board = [];
        this.moveHistory = [];

        this.rewardConfig = {
            win: 10,
            lose: -10,
            draw: 0.5,
            block: 5,
            lineOfThree: 1,
            center: 0.2,
            minEpsilon: 0.10
        };

        this.reset();
    }

    reset() {
        this.board = Array(this.ROWS).fill(0).map(() => Array(this.COLS).fill(0));
        this.moveHistory = [];
    }

    getValidColumns(grid = this.board) {
        const valid = [];
        for (let c = 0; c < this.COLS; c++) {
            if (grid[0][c] === 0) valid.push(c);
        }
        return valid;
    }

    isFull(grid = this.board) {
        return grid[0].every(cell => cell !== 0);
    }

    dropToken(col, player, grid = this.board, recordHistory = true) {
        if (col < 0 || col >= this.COLS || grid[0][col] !== 0) return null;
        for (let r = this.ROWS - 1; r >= 0; r--) {
            if (grid[r][col] === 0) {
                grid[r][col] = player;
                if (recordHistory && grid === this.board) {
                    this.moveHistory.push({ r, c: col, player });
                }
                return { r, c: col };
            }
        }
        return null;
    }

    undoLastExchange() {
        if (this.moveHistory.length < 2) return false;
        for (let i = 0; i < 2; i++) {
            const last = this.moveHistory.pop();
            if (last) this.board[last.r][last.c] = 0;
        }
        return true;
    }

    checkWinner(player, grid = this.board) {
        const R = this.ROWS, C = this.COLS;
        for (let r = 0; r < R; r++)
            for (let c = 0; c < C - 3; c++)
                if (grid[r][c] === player && grid[r][c+1] === player && grid[r][c+2] === player && grid[r][c+3] === player)
                    return [[r, c], [r, c+1], [r, c+2], [r, c+3]];
        for (let r = 0; r < R - 3; r++)
            for (let c = 0; c < C; c++)
                if (grid[r][c] === player && grid[r+1][c] === player && grid[r+2][c] === player && grid[r+3][c] === player)
                    return [[r, c], [r+1, c], [r+2, c], [r+3, c]];
        for (let r = 3; r < R; r++)
            for (let c = 0; c < C - 3; c++)
                if (grid[r][c] === player && grid[r-1][c+1] === player && grid[r-2][c+2] === player && grid[r-3][c+3] === player)
                    return [[r, c], [r-1, c+1], [r-2, c+2], [r-3, c+3]];
        for (let r = 0; r < R - 3; r++)
            for (let c = 0; c < C - 3; c++)
                if (grid[r][c] === player && grid[r+1][c+1] === player && grid[r+2][c+2] === player && grid[r+3][c+3] === player)
                    return [[r, c], [r+1, c+1], [r+2, c+2], [r+3, c+3]];
        return null;
    }

    didBlockOpponentWin(col, opponent, grid = this.board) {
        let r = 0;
        while (r < this.ROWS && grid[r][col] === 0) r++;
        if (r === this.ROWS) return false;
        const original = grid[r][col];
        grid[r][col] = opponent;
        const blocked = this.checkWinner(opponent, grid) !== null;
        grid[r][col] = original;
        return blocked;
    }

    createdLineOfThree(col, player, grid = this.board) {
        let r = 0;
        while (r < this.ROWS && grid[r][col] === 0) r++;
        if (r === this.ROWS) return false;
        let count = 0;
        const checkWindow = (r1, c1, r2, c2, r3, c3, r4, c4) => {
            if (r1 < 0 || r1 >= this.ROWS || c1 < 0 || c1 >= this.COLS ||
                r4 < 0 || r4 >= this.ROWS || c4 < 0 || c4 >= this.COLS) return;
            const w = [grid[r1][c1], grid[r2][c2], grid[r3][c3], grid[r4][c4]];
            if (w.filter(v => v === player).length === 3 && w.filter(v => v === 0).length === 1) count++;
        };
        for (let i = 0; i < 4; i++) {
            checkWindow(r, col-3+i, r, col-2+i, r, col-1+i, r, col+i);
            checkWindow(r+i, col, r+1+i, col, r+2+i, col, r+3+i, col);
            checkWindow(r-3+i, col-3+i, r-2+i, col-2+i, r-1+i, col-1+i, r+i, col+i);
            checkWindow(r-3+i, col+3-i, r-2+i, col+2-i, r-1+i, col+1-i, r+i, col-i);
        }
        return count > 0;
    }

    evaluateMoveReward(col, player, isWin, isDraw) {
        const cfg = this.rewardConfig;
        if (isWin) return { reward: cfg.win, popup: { text: `🏆 Victoire ! (+${cfg.win})`, color: '#2ed573' } };
        if (isDraw) return { reward: cfg.draw, popup: { text: `🤝 Match nul (+${cfg.draw})`, color: '#70a1ff' } };

        let reward = 0;
        let popup = null;
        const opponent = player === 1 ? 2 : 1;

        if (col === 3 && cfg.center > 0) {
            reward += cfg.center;
            popup = { text: `🎯 Centre (+${cfg.center})`, color: '#70a1ff' };
        } else if ((col === 2 || col === 4) && cfg.center > 0) {
            reward += cfg.center / 2;
        }

        if (cfg.lineOfThree > 0 && this.createdLineOfThree(col, player)) {
            reward += cfg.lineOfThree;
            popup = { text: `⚡ Menace de 3 ! (+${cfg.lineOfThree})`, color: '#ffa502' };
        }

        if (cfg.block > 0 && this.didBlockOpponentWin(col, opponent)) {
            reward += cfg.block;
            popup = { text: `🛡️ Parade décisive ! (+${cfg.block})`, color: '#2ed573' };
        }

        return { reward, popup };
    }

    // --- SPARRING PARTNER (BOT PROFESSEUR) ---
    getTeacherMove(player, difficulty = 'GUARDIAN') {
        const valid = this.getValidColumns();
        if (valid.length === 0) return undefined;
        const opponent = player === 1 ? 2 : 1;

        // 1. Gagner immédiatement si possible
        for (const c of valid) {
            const temp = this.cloneBoard();
            this.dropToken(c, player, temp, false);
            if (this.checkWinner(player, temp)) return c;
        }

        // 2. Bloquer la victoire imminente de l'adversaire (Niveau Gardien)
        if (difficulty === 'GUARDIAN') {
            for (const c of valid) {
                const temp = this.cloneBoard();
                this.dropToken(c, opponent, temp, false);
                if (this.checkWinner(opponent, temp)) return c;
            }
        }

        // 3. Jouer au centre par défaut ou au hasard
        if (valid.includes(3) && Math.random() < 0.6) return 3;
        return valid[Math.floor(Math.random() * valid.length)];
    }
}
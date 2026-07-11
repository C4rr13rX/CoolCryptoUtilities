var game = new Phaser.Game(500, 500, Phaser.CANVAS, 'gameContainer', { create: create });

var model = {
    board: [],
    startBoard: [],
    tiles: [],
    moves: 0,
    won: false,
    winOverlay: null,
    moveCountText: null
};

function create() {
    game.stage.backgroundColor = '#34495e';
    
    // Title
    var title = game.add.text(250, 20, 'Lights Out', { font: '24px Arial', fill: '#ecf0f1' });
    title.anchor.set(0.5, 0);
    
    // Move counter
    model.moveCountText = game.add.text(10, 50, 'Moves: 0', { font: '18px Arial', fill: '#ecf0f1' });
    
    // Buttons
    var newPuzzleButton = game.add.text(350, 50, 'New Puzzle', { font: '16px Arial', fill: '#ecf0f1' });
    newPuzzleButton.inputEnabled = true;
    newPuzzleButton.events.onInputDown.add(generateNewPuzzle);
    
    var resetButton = game.add.text(350, 80, 'Reset', { font: '16px Arial', fill: '#ecf0f1' });
    resetButton.inputEnabled = true;
    resetButton.events.onInputDown.add(reset);
    
    // Initialize board and tiles
    for (var row = 0; row < 5; row++) {
        model.board[row] = [];
        model.tiles[row] = [];
        for (var col = 0; col < 5; col++) {
            model.board[row][col] = 0;
            
            var tile = game.add.graphics(100 + col * 60, 100 + row * 60);
            tile.beginFill(0x333333);
            tile.drawRect(0, 0, 50, 50);
            tile.endFill();
            tile.inputEnabled = true;
            tile.events.onInputDown.add((function(r, c) {
                return function() { toggleCell(r, c); };
            })(row, col));
            
            model.tiles[row][col] = tile;
        }
    }
    
    // Keyboard shortcuts
    game.input.keyboard.addKey(Phaser.Keyboard.R).onDown.add(function() {
        if (model.won) generateNewPuzzle();
        else reset();
    });
    
    game.input.keyboard.addKey(Phaser.Keyboard.F).onDown.add(function() {
        game.scale.startFullScreen();
    });
    
    game.input.keyboard.addKey(Phaser.Keyboard.ESC).onDown.add(function() {
        game.scale.stopFullScreen();
    });
    
    generateNewPuzzle();
}

function redrawTile(row, col) {
    var tile = model.tiles[row][col];
    tile.clear();
    tile.beginFill(model.board[row][col] ? 0xffff00 : 0x333333);
    tile.drawRect(0, 0, 50, 50);
    tile.endFill();
}

function toggleCell(row, col) {
    if (model.won) return;
    
    // Toggle clicked cell
    model.board[row][col] = model.board[row][col] ? 0 : 1;
    
    // Toggle adjacent cells
    var directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    for (var i = 0; i < directions.length; i++) {
        var r = row + directions[i][0];
        var c = col + directions[i][1];
        if (r >= 0 && r < 5 && c >= 0 && c < 5) {
            model.board[r][c] = model.board[r][c] ? 0 : 1;
        }
    }
    
    // Update moves and redraw
    model.moves++;
    model.moveCountText.text = 'Moves: ' + model.moves;
    
    // Redraw all affected tiles
    redrawTile(row, col);
    for (var i = 0; i < directions.length; i++) {
        var r = row + directions[i][0];
        var c = col + directions[i][1];
        if (r >= 0 && r < 5 && c >= 0 && c < 5) {
            redrawTile(r, c);
        }
    }
    
    // Check for win
    checkWin();
}

function checkWin() {
    var totalLit = 0;
    for (var row = 0; row < 5; row++) {
        for (var col = 0; col < 5; col++) {
            if (model.board[row][col]) totalLit++;
        }
    }
    
    if (totalLit === 0) {
        model.won = true;
        showWinOverlay();
    }
}

function showWinOverlay() {
    model.winOverlay = game.add.group();
    
    var overlay = game.add.graphics(0, 0);
    overlay.beginFill(0x000000, 0.7);
    overlay.drawRect(0, 0, 500, 500);
    overlay.endFill();
    model.winOverlay.add(overlay);
    
    var winText = game.add.text(250, 200, 'You Win!', { font: '36px Arial', fill: '#ffffff' });
    winText.anchor.set(0.5, 0.5);
    model.winOverlay.add(winText);
    
    var playAgainButton = game.add.text(250, 280, 'Play Again', { font: '24px Arial', fill: '#ffffff' });
    playAgainButton.anchor.set(0.5, 0.5);
    playAgainButton.inputEnabled = true;
    playAgainButton.events.onInputDown.add(function() {
        model.winOverlay.destroy();
        model.winOverlay = null;
        generateNewPuzzle();
    });
    model.winOverlay.add(playAgainButton);
}

function generateNewPuzzle() {
    // Reset board
    for (var row = 0; row < 5; row++) {
        for (var col = 0; col < 5; col++) {
            model.board[row][col] = 0;
        }
    }
    
    // Apply random moves (8-16)
    var moves = 8 + Math.floor(Math.random() * 9);
    for (var i = 0; i < moves; i++) {
        var row = Math.floor(Math.random() * 5);
        var col = Math.floor(Math.random() * 5);
        toggleCellNoCount(row, col);
    }
    
    // Ensure at least one light is on
    var totalLit = 0;
    for (var row = 0; row < 5; row++) {
        for (var col = 0; col < 5; col++) {
            if (model.board[row][col]) totalLit++;
        }
    }
    
    if (totalLit === 0) {
        // Retry if all lights are off
        generateNewPuzzle();
        return;
    }
    
    // Clone to startBoard
    model.startBoard = [];
    for (var row = 0; row < 5; row++) {
        model.startBoard[row] = [];
        for (var col = 0; col < 5; col++) {
            model.startBoard[row][col] = model.board[row][col];
        }
    }
    
    // Reset game state
    model.moves = 0;
    model.won = false;
    model.moveCountText.text = 'Moves: 0';
    
    // Redraw all tiles
    for (var row = 0; row < 5; row++) {
        for (var col = 0; col < 5; col++) {
            redrawTile(row, col);
        }
    }
    
    // Destroy win overlay if exists
    if (model.winOverlay) {
        model.winOverlay.destroy();
        model.winOverlay = null;
    }
}

function toggleCellNoCount(row, col) {
    // Toggle clicked cell
    model.board[row][col] = model.board[row][col] ? 0 : 1;
    
    // Toggle adjacent cells
    var directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    for (var i = 0; i < directions.length; i++) {
        var r = row + directions[i][0];
        var c = col + directions[i][1];
        if (r >= 0 && r < 5 && c >= 0 && c < 5) {
            model.board[r][c] = model.board[r][c] ? 0 : 1;
        }
    }
}

function reset() {
    // Clone startBoard back to board
    for (var row = 0; row < 5; row++) {
        for (var col = 0; col < 5; col++) {
            model.board[row][col] = model.startBoard[row][col];
        }
    }
    
    // Reset game state
    model.moves = 0;
    model.won = false;
    model.moveCountText.text = 'Moves: 0';
    
    // Redraw all tiles
    for (var row = 0; row < 5; row++) {
        for (var col = 0; col < 5; col++) {
            redrawTile(row, col);
        }
    }
    
    // Destroy win overlay if exists
    if (model.winOverlay) {
        model.winOverlay.destroy();
        model.winOverlay = null;
    }
}

function countLit() {
    var count = 0;
    for (var row = 0; row < 5; row++) {
        for (var col = 0; col < 5; col++) {
            if (model.board[row][col]) count++;
        }
    }
    return count;
}

// Expose required functions
window.render_game_to_text = function() {
    return JSON.stringify({
        coordinate_convention: 'row,column; origin top-left; zero-based',
        mode: model.won ? 'won' : 'playing',
        board: model.board,
        moves: model.moves,
        lit_count: countLit()
    });
};

window.advanceTime = function(ms) {
    return ms;
};
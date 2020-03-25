var SAT = require('sat');
var quadtree = require('simple-quadtree')

// Import game settings.
var conf = require('../config.json');
// Import utilities.
var util = require('./lib/util')

var tree = quadtree(0, 0, conf.gameWidth, conf.gameHeight)

var agents = []
var food = []
var ejectedMasses = []
var viruses = []

var leaderboard = []
var leaderboardChanged = false

var V = SAT.Vector;
var C = SAT.Circle;

/*** Updating game state ***/
function addFood(toAdd) {
    var radius = util.massToRadius(conf.foodMass);
    while (toAdd--) {
        var position = conf.foodUniformDisposition ? util.uniformPosition(food, radius) : util.randomPosition(radius);
        food.push({
            // Make IDs unique.
            id: ((new Date()).getTime() + '' + food.length) >>> 0,
            x: position.x,
            y: position.y,
            radius: radius,
            mass: Math.random() + 2,
            hue: Math.round(Math.random() * 360)
        });
    }
}

function removeFood(toRem) {
    while (toRem--) {
        food.pop();
    }
}

function addVirus(toAdd) {
    while (toAdd--) {
        var mass = util.randomInRange(conf.virus.defaultMass.from, conf.virus.defaultMass.to, true);
        var radius = util.massToRadius(mass);
        var position = conf.virusUniformDisposition ? util.uniformPosition(viruses, radius) : util.randomPosition(radius);
        viruses.push({
            id: ((new Date()).getTime() + '' + viruses.length) >>> 0,
            x: position.x,
            y: position.y,
            radius: radius,
            mass: mass,
            fill: conf.virus.fill,
            stroke: conf.virus.stroke,
            strokeWidth: conf.virus.strokeWidth
        });
    }
}

function moveEjectedMass(mass) {
    var deg = Math.atan2(mass.target.y, mass.target.x);
    var deltaY = mass.speed * Math.sin(deg);
    var deltaX = mass.speed * Math.cos(deg);

    mass.speed -= 0.5;
    if (mass.speed < 0) {
        mass.speed = 0;
    }
    if (!isNaN(deltaY)) {
        mass.y += deltaY;
    }
    if (!isNaN(deltaX)) {
        mass.x += deltaX;
    }

    var borderCalc = mass.radius + 5;

    if (mass.x > conf.gameWidth - borderCalc) {
        mass.x = conf.gameWidth - borderCalc;
    }
    if (mass.y > conf.gameHeight - borderCalc) {
        mass.y = conf.gameHeight - borderCalc;
    }
    if (mass.x < borderCalc) {
        mass.x = borderCalc;
    }
    if (mass.y < borderCalc) {
        mass.y = borderCalc;
    }
}

function balanceMass() {
    var totalMass = food.length * conf.foodMass +
        agents
            .map(function(u) {return u.massTotal; })
            .reduce(function(pu, cu) { return pu + cu;}, 0);

    var massDiff = conf.gameMass - totalMass;
    var maxFoodDiff = conf.maxFood - food.length;
    var foodDiff = parseInt(massDiff / conf.foodMass) - maxFoodDiff;
    var foodToAdd = Math.min(foodDiff, maxFoodDiff);
    var foodToRemove = -Math.max(foodDiff, maxFoodDiff);

    if (foodToAdd > 0) {
        //console.log('[DEBUG] Adding ' + foodToAdd + ' food to level!');
        addFood(foodToAdd);
        //console.log('[DEBUG] Mass rebalanced!');
    }
    else if (foodToRemove > 0) {
        //console.log('[DEBUG] Removing ' + foodToRemove + ' food from level!');
        removeFood(foodToRemove);
        //console.log('[DEBUG] Mass rebalanced!');
    }

    var virusToAdd = conf.maxVirus - viruses.length;

    if (virusToAdd > 0) {
        addVirus(virusToAdd);
    }
}

function updateLeaderboard(newLeaderboard, numPlayers) {
    leaderboard = newLeaderboard
    var status = '<span class="title">Leaderboard</span>';
    for (var i = 0; i < leaderboard.length; i++) {
        status += '<br />';
        if (leaderboard[i].id == player.id){
            if(leaderboard[i].name.length !== 0)
                status += '<span class="me">' + (i + 1) + '. ' + leaderboard[i].name + "</span>";
            else
                status += '<span class="me">' + (i + 1) + ". An unnamed cell</span>";
        } else {
            if(leaderboard[i].name.length !== 0)
                status += (i + 1) + '. ' + leaderboard[i].name;
            else
                status += (i + 1) + '. An unnamed cell';
        }
    }
    //status += '<br />Players: ' + numPlayers;
    document.getElementById('status').innerHTML = status;
}


function executeVirusCollision(virusCell) {
    function splitCell(cell) {
        if(cell && cell.mass && cell.mass >= c.defaultPlayerMass*2) {
            cell.mass = cell.mass/2;
            cell.radius = util.massToRadius(cell.mass);
            currentPlayer.cells.push({
                mass: cell.mass,
                x: cell.x,
                y: cell.y,
                radius: cell.radius,
                speed: 25
            });
        }
    }

    if(currentPlayer.cells.length < conf.limitSplit && currentPlayer.massTotal >= conf.defaultPlayerMass * 2) {
        //Split single cell from virus
        if (virusCell) {
          splitCell(currentPlayer.cells[virusCell]);
        }
        else {
          //Split all cells
          if (currentPlayer.cells.length < c.limitSplit && currentPlayer.massTotal >= c.defaultPlayerMass*2) {
              var numMax = currentPlayer.cells.length;
              for(var d=0; d<numMax; d++) {
                  splitCell(currentPlayer.cells[d]);
              }
          }
        }
        currentPlayer.lastSplit = new Date().getTime();
    }
}

function movePlayer(player) {
    var x = 0,y = 0;
    for(var i = 0; i < player.cells.length; i++)
    {
        var target = {
            x: player.x - player.cells[i].x + player.target.x,
            y: player.y - player.cells[i].y + player.target.y
        };
        var dist = Math.sqrt(Math.pow(target.y, 2) + Math.pow(target.x, 2));
        var deg = Math.atan2(target.y, target.x);
        var slowDown = 1;
        if (player.cells[i].speed <= 6.25) {
            slowDown = util.log(player.cells[i].mass, conf.slowBase) - initMassLog + 1;
        }

        var deltaY = player.cells[i].speed * Math.sin(deg)/ slowDown;
        var deltaX = player.cells[i].speed * Math.cos(deg)/ slowDown;

        if (player.cells[i].speed > 6.25) {
            player.cells[i].speed -= 0.5;
        }
        if (dist < (50 + player.cells[i].radius)) {
            deltaY *= dist / (50 + player.cells[i].radius);
            deltaX *= dist / (50 + player.cells[i].radius);
        }
        if (!isNaN(deltaY)) {
            player.cells[i].y += deltaY;
        }
        if (!isNaN(deltaX)) {
            player.cells[i].x += deltaX;
        }
        // Find best solution.
        for(var j = 0; j < player.cells.length; j++) {
            if(j != i && player.cells[i] !== undefined) {
                var distance = Math.sqrt(Math.pow(player.cells[j].y - player.cells[i].y, 2) + Math.pow(player.cells[j].x - player.cells[i].x, 2));
                var radiusTotal = (player.cells[i].radius + player.cells[j].radius);
                if (distance < radiusTotal) {
                    if (player.lastSplit > new Date().getTime() - 1000 * conf.mergeTimer) {
                        if (player.cells[i].x < player.cells[j].x) {
                            player.cells[i].x--;
                        } else if (player.cells[i].x > player.cells[j].x) {
                            player.cells[i].x++;
                        }
                        if (player.cells[i].y < player.cells[j].y) {
                            player.cells[i].y--;
                        } else if ((player.cells[i].y > player.cells[j].y)) {
                            player.cells[i].y++;
                        }
                    }
                    else if (distance < radiusTotal / 1.75) {
                        player.cells[i].mass += player.cells[j].mass;
                        player.cells[i].radius = util.massToRadius(player.cells[i].mass);
                        player.cells.splice(j, 1);
                    }
                }
            }
        }
        if (player.cells.length > i) {
            var borderCalc = player.cells[i].radius / 3;
            if (player.cells[i].x > conf.gameWidth - borderCalc) {
                player.cells[i].x = conf.gameWidth - borderCalc;
            }
            if (player.cells[i].y > conf.gameHeight - borderCalc) {
                player.cells[i].y = conf.gameHeight - borderCalc;
            }
            if (player.cells[i].x < borderCalc) {
                player.cells[i].x = borderCalc;
            }
            if (player.cells[i].y < borderCalc) {
                player.cells[i].y = borderCalc;
            }
            x += player.cells[i].x;
            y += player.cells[i].y;
        }
    }
    player.x = x / player.cells.length;
    player.y = y / player.cells.length;
}

/*** tick player ***/
function tickPlayer(currentPlayer) {
    // TODO: do new heartbeat check
    // if(currentPlayer.lastHeartbeat < new Date().getTime() - c.maxHeartbeatInterval) {
    //     sockets[currentPlayer.id].emit('kick', 'Last heartbeat received over ' + c.maxHeartbeatInterval + ' ago.');
    //     sockets[currentPlayer.id].disconnect();
    // }
    movePlayer(currentPlayer);

    function checkFoodCollision(f) {
        return SAT.pointInCircle(new V(f.x, f.y), playerCircle);
    }

    function deleteFood(f) {
        food[f] = {};
        food.splice(f, 1);
    }

    function eatMass(m) {
        if (SAT.pointInCircle(new V(m.x, m.y), playerCircle)) {
            if (m.id == currentPlayer.id && m.speed > 0 && z == m.num)
                return false;
            if (currentCell.mass > m.masa * 1.1)
                return true;
        }
        return false;
    }

    function checkAgentCollision(user) {
        var playerCollisions = []
        for(var i=0; i<user.cells.length; i++) {
            if(user.cells[i].mass > 10 && user.id !== currentPlayer.id) {
                var response = new SAT.Response();
                var collided = SAT.testCircleCircle(playerCircle,
                    new C(new V(user.cells[i].x, user.cells[i].y), user.cells[i].radius),
                    response);
                if (collided) {
                    response.aUser = currentCell;
                    response.bUser = {
                        id: user.id,
                        name: user.name,
                        x: user.cells[i].x,
                        y: user.cells[i].y,
                        num: i,
                        mass: user.cells[i].mass
                    };
                    playerCollisions.push(response);
                }
            }
        }
        return playerCollisions;
    }

    function executeAgentCollision(collision) {
        if (collision.aUser.mass > collision.bUser.mass * 1.1  && collision.aUser.radius > Math.sqrt(Math.pow(collision.aUser.x - collision.bUser.x, 2) + Math.pow(collision.aUser.y - collision.bUser.y, 2))*1.75) {
            console.log('[DEBUG] Killing user: ' + collision.bUser.id);
            console.log('[DEBUG] Collision info:');
            console.log(collision);

            var numAgent = util.findIndex(agents, collision.bUser.id);
            if (numAgent > -1) {
                if (agents[numAgent].cells.length > 1) {
                    agents[numAgent].massTotal -= collision.bUser.mass;
                    agents[numAgent].cells.splice(collision.bUser.num, 1);
                } else {
                    agents.splice(numAgent, 1);
                    //io.emit('playerDied', { name: collision.bUser.name });
                    //sockets[collision.bUser.id].emit('RIP');
                    client.RIP(collision.bUser.id)
                }
            }
            currentPlayer.massTotal += collision.bUser.mass;
            collision.aUser.mass += collision.bUser.mass;
        }
    }

    for (var z = 0; z < currentPlayer.cells.length; z++) {
        var currentCell = currentPlayer.cells[z];
        var playerCircle = new C(
            new V(currentCell.x, currentCell.y),
            currentCell.radius
        );

        var foodEaten = food.map(checkFoodCollision)
            .reduce(function(a, b, c) { return b ? a.concat(c) : a; }, []);

        foodEaten.forEach(deleteFood);

        var massEaten = ejectedMasses.map(eatMass)
            .reduce(function(a, b, c) {return b ? a.concat(c) : a; }, []);

        var virusCollision = viruses.map(checkFoodCollision)
           .reduce(function(a, b, c) { return b ? a.concat(c) : a; }, []);

        if (virusCollision > 0 && currentCell.mass > viruses[virusCollision].mass) {
          //sockets[currentPlayer.id].emit('virusSplit', z);
          reeinvar = false;
          executeVirusCollision(z);
          viruses.splice(virusCollision, 1);
        }

        var massGained = 0;
        for(var i = 0; i < massEaten.length; i++) {
            massGained += ejectedMasses[massEaten[i]].mass;
            ejectedMasses[massEaten[i]] = {};
            ejectedMasses.splice(massEaten[i], 1);
            for(var j = 0; j < massEaten.length; j++) {
                if(massEaten[i] < massEaten[j]) {
                    massEaten[j]--;
                }
            }
        }

        if (typeof(currentCell.speed) == "undefined")
            currentCell.speed = 6.25;
        massGained += (foodEaten.length * conf.foodMass);
        currentCell.mass += massGained;
        currentPlayer.massTotal += massGained;
        currentCell.radius = util.massToRadius(currentCell.mass);
        playerCircle.r = currentCell.radius;

        tree.clear();
        agents.forEach(tree.put);
        var playerCollisions = tree.get(currentPlayer, checkAgentCollision);

        playerCollisions.forEach(executeAgentCollision);
    }
}

/*** Running the game ***/
function moveloop() {
    for (var i = 0; i < agents.length; i++) {
        tickPlayer(agents[i]);
    }
    for (i = 0; i < ejectedMasses.length; i++) {
        if (ejectedMasses[i].speed > 0) {
            moveEjectedMass(ejectedMasses[i]);
        }
    }
}

function gameloop() {
    if (agents.length > 0) {
        // updates the leaderboard
        agents.sort( function(a, b) { return b.massTotal - a.massTotal; });
        var topAgents = [];

        for (var i = 0; i < Math.min(10, agents.length); i++) {
            if(agents[i].type == 'player') {
                topAgents.push({
                    id: agents[i].id,
                    name: agents[i].name
                });
            }
        }
        if (isNaN(leaderboard) || leaderboard.length !== topAgents.length) {
            leaderboard = topAgents;
            leaderboardChanged = true;
        }
        else {
            for (i = 0; i < leaderboard.length; i++) {
                if (leaderboard[i].id !== topAgents[i].id) {
                    leaderboard = topAgents;
                    leaderboardChanged = true;
                    break;
                }
            }
        }

        // blobs lose mass over time according to this rule
        for (i = 0; i < agents.length; i++) {
            for(var z=0; z < agents[i].cells.length; z++) {
                if (agents[i].cells[z].mass * (1 - (conf.massLossRate / 1000)) > conf.defaultPlayerMass && agents[i].massTotal > conf.minMassLoss) {
                    var massLoss = agents[i].cells[z].mass * (1 - (conf.massLossRate / 1000));
                    agents[i].massTotal -= agents[i].cells[z].mass - massLoss;
                    agents[i].cells[z].mass = massLoss;
                }
            }
        }
    }

    // make sure that the total mass in the game is balanced (between food, blobs, etc.)
    balanceMass();
}

function updateAgentViews() {
    agents.forEach(function(u) {
        // center the view if x/y is undefined, this will happen for spectators
        u.x = u.x || conf.gameWidth / 2;
        u.y = u.y || conf.gameHeight / 2;

        var visibleFood = food
            .map(function(f) {
                if (f.x > u.x - u.screenWidth/2 - 20 &&
                    f.x < u.x + u.screenWidth/2 + 20 &&
                    f.y > u.y - u.screenHeight/2 - 20 &&
                    f.y < u.y + u.screenHeight/2 + 20) {
                    return f;
                }
            })
            .filter(function(f) { return f; });

        var visibleViruses = viruses
            .map(function(f) {
                if ( f.x > u.x - u.screenWidth/2 - f.radius &&
                    f.x < u.x + u.screenWidth/2 + f.radius &&
                    f.y > u.y - u.screenHeight/2 - f.radius &&
                    f.y < u.y + u.screenHeight/2 + f.radius) {
                    return f;
                }
            })
            .filter(function(f) { return f; });

        var visibleFiredMass = ejectedMasses
            .map(function(f) {
                if ( f.x+f.radius > u.x - u.screenWidth/2 - 20 &&
                    f.x-f.radius < u.x + u.screenWidth/2 + 20 &&
                    f.y+f.radius > u.y - u.screenHeight/2 - 20 &&
                    f.y-f.radius < u.y + u.screenHeight/2 + 20) {
                    return f;
                }
            })
            .filter(function(f) { return f; });

        var visibleAgents = agents
            .map(function(f) {
                for(var z=0; z<f.cells.length; z++)
                {
                    if ( f.cells[z].x+f.cells[z].radius > u.x - u.screenWidth/2 - 20 &&
                        f.cells[z].x-f.cells[z].radius < u.x + u.screenWidth/2 + 20 &&
                        f.cells[z].y+f.cells[z].radius > u.y - u.screenHeight/2 - 20 &&
                        f.cells[z].y-f.cells[z].radius < u.y + u.screenHeight/2 + 20) {
                        z = f.cells.lenth;
                        if(f.id !== u.id) {
                            return {
                                id: f.id,
                                x: f.x,
                                y: f.y,
                                cells: f.cells,
                                massTotal: Math.round(f.massTotal),
                                hue: f.hue,
                                name: f.name
                            };
                        } else {
                            return {
                                x: f.x,
                                y: f.y,
                                cells: f.cells,
                                massTotal: Math.round(f.massTotal),
                                hue: f.hue,
                            };
                        }
                    }
                }
            })
            .filter(function(f) { return f; });

        //sockets[u.id].emit('serverTellPlayerMove', visibleCells, visibleFood, visibleMass, visibleVirus);
        u.visibleAgents = visibleAgents;
        u.visibleFood = visibleFood;
        u.visibleFiredMass = visibleFiredMass;
        u.visibleViruses = visibleViruses;

        if (leaderboardChanged) {
            updateLeaderboard(leaderboard, agents.length);
        }
    });
    leaderboardChanged = false;
}

setInterval(moveloop, 1000 / 60);
setInterval(gameloop, 1000);
setInterval(updateAgentViews, 1000 / conf.networkUpdateFactor);
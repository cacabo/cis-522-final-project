var foodConfig = {
    border: 0,
};

var playerConfig = {
    border: 6,
    textColor: '#FFFFFF',
    textBorder: '#000000',
    textBorderSize: 3,
    defaultSize: 30
};

class Player {
    constructor() {
        this.id = -1;
        this.x = global.screenWidth / 2;
        this.y = global.screenHeight / 2;
        this.screenWidth = global.screenWidth;
        this.screenHeight = global.screenHeight;
        this.target = {x: global.screenWidth / 2, y: global.screenHeight / 2}
        this.visibleFood = [];
        this.visibleViruses = [];
        this.visibleFiredMass = [];
        this.visibleAgents = [];
        this.leaderboard = [];
    }


}
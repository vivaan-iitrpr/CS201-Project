#include "raylib.h"
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <string>
#include <limits>
#include <sstream>

using namespace std;

const int MAZE_WIDTH = 20;
const int MAZE_HEIGHT = 22;
const int CELL_SIZE = 30;
const int WIDTH = MAZE_WIDTH * CELL_SIZE;
const int HEIGHT = MAZE_HEIGHT * CELL_SIZE;
const int INF = numeric_limits<int>::max(); 

enum CellType { EMPTY = 0, WALL = 1, FOOD = 2 };

const int MAZE[MAZE_HEIGHT][MAZE_WIDTH] = {
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1},
    {1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1},
    {1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1},
    {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1},
    {1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1},
    {1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1},
    {1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1},
    {0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0},
    {1, 1, 1, 1, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 1, 1, 1, 1},
    {2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 2, 2, 2, 2},
    {1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1},
    {0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0},
    {1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1},
    {1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1},
    {1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1},
    {1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1},
    {1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1},
    {1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1},
    {1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1},
    {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
};


// SPeed controls 
// 60 FPS : SETTARGETFPS
// GHPST MOVES EVERY 25 FRAMES (BY GHOST MOVE DELAY)
// DUNGEON IN AUTO SOLVER MOVER EVERY FRAME -> THERE IS NO DELAY IN FRAME PUT 

// Algorithms 
//Algorithm changes take effect immediately during gameplay - pressing A, D, or F,
//  the ghosts will immediately start using the new pathfinding algorithm to chase

//Auto-solver (Dungeon):->  Uses A* algorithm (find_escape_path)
//ALgorithm for Ghost can be changed by A D or F 



struct Position {   // just defined two operations == and != on the positions
    int row, col;
    bool operator==(const Position& other) const {
        return row == other.row && col == other.col;    
    }
    bool operator!=(const Position& other) const {
        return !(*this == other);
    }
};

namespace std {
    template<>
    struct hash<Position> {
        size_t operator()(const Position& p) const {
            return hash<int>()(p.row) ^ hash<int>()(p.col);  // generated a new hashed value by xor using the hash stl 
        }
    };
}

const vector<Position> DIRECTIONS = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};  //4 dirn
const Position DUNGEON_START = {1, 1};
const vector<Position> GHOST_STARTS = {{10, 9}, {10, 10}, {9, 9}, {9, 10}};

class DungeonGame {
private:
    Position Dungeon;
    vector<Position> ghosts;
    int score;
    int maze[MAZE_HEIGHT][MAZE_WIDTH];
    string algorithm;
    Font font;
    int ghost_move_counter; //counts ghost movements.
    const int GHOST_MOVE_DELAY = 25; // GHOST MOVEMENT FREQ IS CONTROLLED 
    bool auto_solve;
    // vector<Position> safe_path;
    vector<vector<vector<int>>> floyd_dist;
    vector<vector<vector<Position>>> floyd_next;
    vector<Position> game_analysis_path;
    bool game_over;

    Position find_nearest_food() {
        // just iterates over all the cells and find the min distance of any food avialable
        Position nearest_food = Dungeon;
        int min_distance = INF;
        
        for (int i = 0; i < MAZE_HEIGHT; i++) {
            for (int j = 0; j < MAZE_WIDTH; j++) {
                if (maze[i][j] == FOOD) {
                    Position food_pos = {i, j};
                    int dist = floyd_dist[Dungeon.row][Dungeon.col][i * MAZE_WIDTH + j];
                    if (dist < min_distance && dist != INF) {
                        min_distance = dist;
                        nearest_food = food_pos;
                    }
                }
            }
        }
        return nearest_food; 
    }

    bool is_move_safe(const Position& pos, int safety_threshold = 3) {
        for (const auto& ghost : ghosts) {
            int ghost_dist = floyd_dist[pos.row][pos.col][ghost.row * MAZE_WIDTH + ghost.col];
            if (ghost_dist != INF && ghost_dist < safety_threshold) {  // IE IT WILL CATCH IT 
                return false;
            }
        }
        return true;
    }

    vector<Position> find_safe_path_to_food() {
        Position target_food = find_nearest_food();
        if (target_food == Dungeon) return {};  //No food-founded 

        // TRY In different safety thresholds by using find_escape_path function, if we can escape 
        for (int safety = 4; safety >= 2; safety--) {
            vector<Position> path = find_escape_path(target_food, safety);
            if (!path.empty()) {
                return path;
            }
        }
        return {};
    }

    vector<Position> find_escape_path(const Position& target, int safety_threshold) {
        // tehrefore A* hi use kar rahe for Auto Solver
        auto cmp = [](const pair<int, Position>& left, const pair<int, Position>& right) { // custom comparator func
            return left.first > right.first; //compares based on first value being greater
        };
        priority_queue<pair<int, Position>, vector<pair<int, Position>>, decltype(cmp)> pq(cmp); // based on min diostance 
        pq.push({0, Dungeon});

        unordered_map<Position, Position> came_from; // for path reconstruction we will require previous position
        unordered_map<Position, int> g_score;
        g_score[Dungeon] = 0; // the actual cost or 

        while (!pq.empty()) {
            Position current = pq.top().second; // the position with the lowest cost 
            pq.pop();

            if (current == target){ // if target reached then just reconstruct the path 
                return reconstruct_path(came_from, current);
            }

            for (const auto& direction : DIRECTIONS) {
                Position next = {current.row + direction.row, current.col + direction.col};
                if (is_valid_move(next) && is_move_safe(next, safety_threshold)) {
                    int new_cost = g_score[current] + 1; // finding the new score 
                    if (g_score.find(next) == g_score.end() || new_cost < g_score[next]) { // if score was not found ie first time vis/less cost now 
                                                                //then update the variab;es
                        came_from[next] = current;
                        g_score[next] = new_cost;
                        int priority = new_cost + heuristic(next, target); // heuristic is just the sum of the x and y coordinates of)  : the Manhattan Distance  
                        pq.push({priority, next});
                    }
                }
            }
        }
        return {};
    }

    Position get_emergency_escape() { //"maximizing the minimum distance" strategy
        int max_min_dist = -1; // nearest ghost ka distance ka max -> will give the best place to go 
        Position best_move = Dungeon;
        
        for (const auto& dir : DIRECTIONS) {
            Position next = {Dungeon.row + dir.row, Dungeon.col + dir.col};
            if (is_valid_move(next)) {
                int min_ghost_dist = INF;
                for (const auto& ghost : ghosts) {
                    int dist = floyd_dist[next.row][next.col][ghost.row * MAZE_WIDTH + ghost.col];
                    if (dist != INF) {
                        min_ghost_dist = min(min_ghost_dist, dist);
                    }
                }
                if (min_ghost_dist > max_min_dist) {
                    max_min_dist = min_ghost_dist;
                    best_move = next;
                }
            }
        }
        return best_move;
    }


    void initialize_floyd_warshall() {
        int V = MAZE_HEIGHT * MAZE_WIDTH;
        floyd_dist.resize(MAZE_HEIGHT, vector<vector<int>>(MAZE_WIDTH, vector<int>(V, INF))); // ditance b/w all pairs
        floyd_next.resize(MAZE_HEIGHT, vector<vector<Position>>(MAZE_WIDTH, vector<Position>(V, {-1, -1}))); // 
        // row,col,ending_position (flattened to 1d by multiplying with maze width)

        // Initialize distances
        for (int i = 0; i < MAZE_HEIGHT; i++) {
            for (int j = 0; j < MAZE_WIDTH; j++) {
                if (maze[i][j] != WALL) {
                    for (const auto& dir : DIRECTIONS) {
                        int ni = i + dir.row;
                        int nj = j + dir.col;
                        if (is_valid_move({ni, nj})) {
                            floyd_dist[i][j][ni * MAZE_WIDTH + nj] = 1;
                            floyd_next[i][j][ni * MAZE_WIDTH + nj] = {ni, nj}; // for directly connected cells udpate the distance 
                        }
                    }
                    floyd_dist[i][j][i * MAZE_WIDTH + j] = 0;
                    floyd_next[i][j][i * MAZE_WIDTH + j] = {i, j}; // for itself
                }
            }
        }

        // Floyd-Warshall algorithm
        for (int k = 0; k < V; k++) {
            // k is the intermediate node ie it will contain the coordibates flattened into one number  
            for (int i = 0; i < MAZE_HEIGHT; i++) {
                for (int j = 0; j < MAZE_WIDTH; j++) {
                    if (maze[i][j] == WALL) continue;
                    for (int u = 0; u < V; u++) { 
                        int kr = k / MAZE_WIDTH;
                        int kc = k % MAZE_WIDTH;

                        //converts 1d k to coordinates 

                        int ur = u / MAZE_WIDTH;
                        int uc = u % MAZE_WIDTH;
                        
                        if (floyd_dist[i][j][k] != INF && floyd_dist[kr][kc][u] != INF) {
                            int new_dist = floyd_dist[i][j][k] + floyd_dist[kr][kc][u];
                            if (new_dist < floyd_dist[i][j][u]) {
                                floyd_dist[i][j][u] = new_dist;
                                floyd_next[i][j][u] = floyd_next[i][j][k];
                                // updates distances if its smaller
                                // and keep on maintaing ki kha se jaana hoga 
                            }
                        }
                    }
                }
            }
        }
    }

    vector<Position> get_floyd_path(const Position& start, const Position& end) {
        // RECONSTRUCTS THE Actual path between two positions ie to get the seq of moves to reach destinatuon
        vector<Position> path;
        if (floyd_dist[start.row][start.col][end.row * MAZE_WIDTH + end.col] == INF) {
            return path;
        }

        Position current = start;
        while (current != end) {
            path.push_back(current);
            current = floyd_next[current.row][current.col][end.row * MAZE_WIDTH + end.col];
        }
        path.push_back(end);
        return path;
    }
Position find_safest_move() {

        static vector<Position> current_path;
        static int path_index = 0;

        // If we have no path or have reached the end of current path, calculate new path
        //  current_path remebers the Planned paTh between cells
        if (current_path.empty() || path_index >= current_path.size()) {
            current_path = find_safe_path_to_food();
            path_index = 0;
            
            // If no safe path to food is found, use emergency escape
            if (current_path.empty()) {
                return get_emergency_escape();
            }
        }

        // Check if current path is still safe
        if (path_index < current_path.size()) {
            Position next_pos = current_path[path_index];
            if (!is_move_safe(next_pos)) {
                // Path became unsafe, recalculate
                current_path = find_safe_path_to_food();
                path_index = 0;
                if (current_path.empty()) {
                    return get_emergency_escape();
                }
                next_pos = current_path[path_index];
            }
            path_index++;
            return next_pos;
        }

        return get_emergency_escape();
    }

    bool is_valid_move(const Position& pos) const {
        return pos.row >= 0 && pos.row < MAZE_HEIGHT &&
               pos.col >= 0 && pos.col < MAZE_WIDTH &&
               maze[pos.row][pos.col] != WALL;
    }
    
    vector<Position> reconstruct_path(const unordered_map<Position, Position>& came_from, Position current) {
        // gives the path by which dungeon goes by using parent 
        vector<Position> path;
        while (came_from.find(current) != came_from.end()) {
            path.push_back(current);
            current = came_from.at(current);
        }
        reverse(path.begin(), path.end());
        return path;
    }

    int heuristic(const Position& a, const Position& b) {
        return abs(a.row - b.row) + abs(a.col - b.col);
    }

    vector<Position> a_star(const Position& start, const Position& goal) {
        auto cmp = [](const pair<int, Position>& left, const pair<int, Position>& right) {
            return left.first > right.first;
        };
        priority_queue<pair<int, Position>, vector<pair<int, Position>>, decltype(cmp)> pq(cmp);
        pq.push({0, start});

        unordered_map<Position, Position> came_from;
        unordered_map<Position, int> g_score; // cost from  START 
        unordered_map<Position, int> f_score; // estimated total cost 

        g_score[start] = 0;
        f_score[start] = heuristic(start, goal);

        while (!pq.empty()) {
            Position current = pq.top().second;
            pq.pop();

            if (current == goal) {
                return reconstruct_path(came_from, current);
            }

            for (const auto& direction : DIRECTIONS) {
                Position neighbor = {current.row + direction.row, current.col + direction.col};
                if (is_valid_move(neighbor)) {
                    int tentative_g_score = g_score[current] + 1;
                    if (g_score.find(neighbor) == g_score.end() || tentative_g_score < g_score[neighbor]) {
                        came_from[neighbor] = current;
                        g_score[neighbor] = tentative_g_score;
                        f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal);
                        pq.push({f_score[neighbor], neighbor});
                    }
                }
            }
        }
        return {};
    }

    vector<Position> dijkstra(const Position& start, const Position& goal) {
        queue<pair<int, Position>> queue;
        queue.push({0, start});

        unordered_map<Position, Position> came_from;
        unordered_map<Position, int> cost_so_far;
        cost_so_far[start] = 0;

        while (!queue.empty()) {
            Position current = queue.front().second;
            queue.pop();

            if (current == goal) {
                return reconstruct_path(came_from, current);
            }

            for (const auto& direction : DIRECTIONS) {
                Position next = {current.row + direction.row, current.col + direction.col};
                if (is_valid_move(next)) {
                    int new_cost = cost_so_far[current] + 1;
                    if (cost_so_far.find(next) == cost_so_far.end() || new_cost < cost_so_far[next]) {
                        cost_so_far[next] = new_cost;
                        queue.push({new_cost, next});
                        came_from[next] = current;
                    }
                }
            }
        }
        return {};
    }

    vector<Position> find_path(const Position& start, const Position& goal) {
        if (algorithm == "floyd_warshall") {
            return get_floyd_path(start, goal);
        } else if (algorithm == "a_star") {
            return a_star(start, goal);
        } else if (algorithm == "dijkstra") {
            return dijkstra(start, goal);
        }
        return {};
    }

public:
    DungeonGame() : score(0), algorithm("a_star"), ghost_move_counter(0), auto_solve(false), game_over(false) {
        reset_game();
        // font = LoadFont("resources/font.ttf");
        initialize_floyd_warshall();
    }


    void reset_game() {
        Dungeon = DUNGEON_START;
        ghosts = GHOST_STARTS;
        score = 0;
        memcpy(maze, MAZE, sizeof(MAZE));
        game_over = false;
        game_analysis_path.clear();
        // safe_path.clear();
    }

    void move_Dungeon(int drow, int dcol) {
        Position new_pos = {Dungeon.row + drow, Dungeon.col + dcol};
        if (is_valid_move(new_pos)) {
            if (maze[new_pos.row][new_pos.col] == FOOD) {
                score += 10;
                maze[new_pos.row][new_pos.col] = EMPTY;
            }
            Dungeon = new_pos;
            game_analysis_path.push_back(new_pos);
        }
    }

    void move_ghosts() {
        ghost_move_counter++;
        if (ghost_move_counter >= GHOST_MOVE_DELAY) {
            for (auto& ghost : ghosts) {
                vector<Position> path = find_path(ghost, Dungeon);
                if (!path.empty() && path.size() > 1) {
                    ghost = path[1];
                }
            }
            ghost_move_counter = 0;
        }
    }

    void auto_move() {
        if (auto_solve) {
            Position next_pos = find_safest_move();
            move_Dungeon(next_pos.row - Dungeon.row, next_pos.col - Dungeon.col);
        }
    }

    bool check_collision() const {
        return find(ghosts.begin(), ghosts.end(), Dungeon) != ghosts.end();
    }

    bool check_win_condition() const {
        for (int i = 0; i < MAZE_HEIGHT; ++i) {
            for (int j = 0; j < MAZE_WIDTH; ++j) {
                if (maze[i][j] == FOOD) {
                    return false;
                }
            }
        }
        return true;
    }

    void draw() {
        ClearBackground(BLACK);

        // Draw maze
        for (int i = 0; i < MAZE_HEIGHT; ++i) {
            for (int j = 0; j < MAZE_WIDTH; ++j) {
                Rectangle rect = {static_cast<float>(j * CELL_SIZE),
                                static_cast<float>(i * CELL_SIZE),
                                static_cast<float>(CELL_SIZE),
                                static_cast<float>(CELL_SIZE)};
                if (maze[i][j] == WALL) {
                    DrawRectangleRec(rect, BLUE);
                } else if (maze[i][j] == FOOD) {
                    DrawCircle(j * CELL_SIZE + CELL_SIZE / 2,
                             i * CELL_SIZE + CELL_SIZE / 2,
                             CELL_SIZE / 10, WHITE);
                }
            }
        }

        // Draw path analysis if game is over
        if (game_over && !game_analysis_path.empty()) {
            for (size_t i = 1; i < game_analysis_path.size(); ++i) {
                DrawLine(
                    game_analysis_path[i-1].col * CELL_SIZE + CELL_SIZE / 2,
                    game_analysis_path[i-1].row * CELL_SIZE + CELL_SIZE / 2,
                    game_analysis_path[i].col * CELL_SIZE + CELL_SIZE / 2,
                    game_analysis_path[i].row * CELL_SIZE + CELL_SIZE / 2,
                    GREEN
                );
            }
        }


        // // Drawin safe path for auto solver
        // if (auto_solve && !safe_path.empty()) {
        //     for (size_t i = 1; i < safe_path.size(); ++i) {
        //         DrawLine(
        //             safe_path[i-1].col * CELL_SIZE + CELL_SIZE / 2,
        //             safe_path[i-1].row * CELL_SIZE + CELL_SIZE / 2,
        //             safe_path[i].col * CELL_SIZE + CELL_SIZE / 2,
        //             safe_path[i].row * CELL_SIZE + CELL_SIZE / 2,
        //             GREEN
        //         );
        //     }
        // }

        // Draw characters
        DrawCircle(Dungeon.col * CELL_SIZE + CELL_SIZE / 2,
                  Dungeon.row * CELL_SIZE + CELL_SIZE / 2,
                  CELL_SIZE / 2, YELLOW);

        for (const auto& ghost : ghosts) {
            DrawCircle(ghost.col * CELL_SIZE + CELL_SIZE / 2,
                      ghost.row * CELL_SIZE + CELL_SIZE / 2,
                      CELL_SIZE / 2, RED);
        }

        // Draw UI
        DrawTextEx(font, TextFormat("Score: %d", score), {10, 10}, 20, 1, WHITE);
        DrawTextEx(font, TextFormat("Algorithm: %s", algorithm.c_str()), {10, 40}, 20, 1, WHITE);
        DrawTextEx(font, TextFormat("Auto-solve: %s", auto_solve ? "ON" : "OFF"), {10, 70}, 20, 1, WHITE);
    }

    void handle_input() {
        if (!auto_solve) {
            if (IsKeyPressed(KEY_UP)) move_Dungeon(-1, 0);
            if (IsKeyPressed(KEY_DOWN)) move_Dungeon(1, 0);
            if (IsKeyPressed(KEY_LEFT)) move_Dungeon(0, -1);
            if (IsKeyPressed(KEY_RIGHT)) move_Dungeon(0, 1);
        }
        if (IsKeyPressed(KEY_A)) algorithm = "a_star";
        if (IsKeyPressed(KEY_D)) algorithm = "dijkstra";
        if (IsKeyPressed(KEY_F)) algorithm = "floyd_warshall";
        if (IsKeyPressed(KEY_SPACE)) auto_solve = !auto_solve;
    }

    void display_analysis() {
        BeginDrawing();
        ClearBackground(BLACK);
        
        stringstream ss;
        ss << "Game Analysis\n\n";
        ss << "Algorithm used: " << algorithm << "\n\n";
        ss << "Path taken by Dungeon:\n";
        
        // Format path with line breaks every 5 positions
        int count = 0;
        for (const auto& pos : game_analysis_path) {
            ss << "(" << pos.row << "," << pos.col << ") ";
            count++;
            if (count % 5 == 0) ss << "\n";
            else ss << "-> ";
        }
        
        ss << "\n\nBest safe paths from current position:\n";
        auto safe_moves = find_safest_move();
        ss << "Recommended next move: (" << safe_moves.row << "," << safe_moves.col << ")\n";
        
        string analysis = ss.str();
        
        // Draw text with smaller font size and proper spacing
        Vector2 textPos = {20, 20};
        float fontSize = 16;
        float lineSpacing = 1.5;
        
        // Split text into lines and draw each line
        string line;
        stringstream analysis_stream(analysis);
        while (getline(analysis_stream, line)) {
            DrawTextEx(font, line.c_str(), textPos, fontSize, 1, WHITE);
            textPos.y += fontSize * lineSpacing;
        }
        
        // Draw continuation prompt at bottom
        DrawTextEx(font, "Press ENTER to play again or ESC to quit", 
                {20, HEIGHT - 40}, fontSize, 1, WHITE);
        EndDrawing();
    }

    void run() {
        InitWindow(WIDTH, HEIGHT, "Dungeon Pathfinding");
        SetTargetFPS(60);

        while (!WindowShouldClose()) {
            if (!game_over) {
                handle_input();
                if (auto_solve) {
                    auto_move();
                }
                move_ghosts();

                if (check_collision() || check_win_condition()) {
                    game_over = true;
                }

                BeginDrawing();
                draw();
                EndDrawing();
            } else {
                display_analysis();
                
                if (IsKeyPressed(KEY_ENTER)) {
                    reset_game();
                } else if (IsKeyPressed(KEY_ESCAPE)) {
                    break;
                }
            }
        }

        CloseWindow();
    }
};

int main() {
    DungeonGame game;
    game.run();
    return 0;
}

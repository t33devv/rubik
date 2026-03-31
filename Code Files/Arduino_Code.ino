// ── Pin Definitions ──────────────────────────────────────────
#define R_STEP 2
#define R_DIR  3
#define U_STEP 12
#define U_DIR  13
#define L_STEP 6
#define L_DIR  7
#define D_STEP 10
#define D_DIR  11
#define B_STEP 8
#define B_DIR  9
#define F_STEP 4
#define F_DIR  5


// ── Motion Config ─────────────────────────────────────────────
#define STEPS_PER_90     50
#define STEP_DELAY_US    500   // ↓ from 1000 — faster step pulse
#define MOVE_PAUSE_MS    80   // ↓ from 400 — shorter settle time between moves


// ── Setup ─────────────────────────────────────────────────────
void setup() {
  delay(5000);
  Serial.begin(9600);
  int stepPins[] = {R_STEP, U_STEP, L_STEP, D_STEP, F_STEP, B_STEP};
  int dirPins[]  = {R_DIR,  U_DIR,  L_DIR,  D_DIR,  F_DIR,  B_DIR};
  for (int i = 0; i < 6; i++) {
    pinMode(stepPins[i], OUTPUT);
    pinMode(dirPins[i],  OUTPUT);
    digitalWrite(dirPins[i], LOW);
    digitalWrite(stepPins[i], LOW);
  }
  delay(3000);
  Serial.println("Starting solve...");
}


// ── Core Rotate Helper ────────────────────────────────────────
void rotateFace(int stepPin, int dirPin, bool clockwise) {
  digitalWrite(dirPin, clockwise ? HIGH : LOW);
  delayMicroseconds(5);
  for (int i = 0; i < STEPS_PER_90; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(STEP_DELAY_US);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(STEP_DELAY_US);
  }
}


// ── Face Move Functions ───────────────────────────────────────
void moveR(bool clockwise = true) { rotateFace(R_STEP, R_DIR,  clockwise); }
void moveU(bool clockwise = true) { rotateFace(U_STEP, U_DIR, !clockwise); }
void moveL(bool clockwise = true) { rotateFace(L_STEP, L_DIR,  clockwise); }
void moveD(bool clockwise = true) { rotateFace(D_STEP, D_DIR, !clockwise); }
void moveF(bool clockwise = true) { rotateFace(F_STEP, F_DIR, !clockwise); }
void moveB(bool clockwise = true) { rotateFace(B_STEP, B_DIR,  clockwise); }


// ── Double Move Functions (180°) — no pause between the two 90° turns ──
void moveR2() { moveR(); moveR(); }
void moveU2() { moveU(); moveU(); }
void moveL2() { moveL(); moveL(); }
void moveD2() { moveD(); moveD(); }
void moveF2() { moveF(); moveF(); }
void moveB2() { moveB(); moveB(); }

void solve() {
  Serial.println("--- Solving ---");
  Serial.println("F2");  moveF2();       delay(MOVE_PAUSE_MS);
  Serial.println("U2");  moveU2();       delay(MOVE_PAUSE_MS);
  Serial.println("R");   moveR();        delay(MOVE_PAUSE_MS);
  Serial.println("B2");  moveB2();       delay(MOVE_PAUSE_MS);
  Serial.println("D'");  moveD(false);   delay(MOVE_PAUSE_MS);
  Serial.println("L2");  moveL2();       delay(MOVE_PAUSE_MS);
  Serial.println("F'");  moveF(false);   delay(MOVE_PAUSE_MS);
  Serial.println("U");   moveU();        delay(MOVE_PAUSE_MS); 
  Serial.println("R2");  moveR2();       delay(MOVE_PAUSE_MS);
  Serial.println("B");   moveB();        delay(MOVE_PAUSE_MS);
  Serial.println("D2");  moveD2();       delay(MOVE_PAUSE_MS);
  Serial.println("L'");  moveL(false);   delay(MOVE_PAUSE_MS);
  Serial.println("F");   moveF();        delay(MOVE_PAUSE_MS);
  Serial.println("U'");  moveU(false);   delay(MOVE_PAUSE_MS);
  Serial.println("R'");  moveR(false);   delay(MOVE_PAUSE_MS);
  Serial.println("=== Solve Complete! ===");
}


// ── Main Loop ─────────────────────────────────────────────────
void loop() {
  solve();
  while (true);
}
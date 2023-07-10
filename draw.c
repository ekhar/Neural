#import "draw.h"
#include "mnist.h"
#import "raylib.h"
#import "rlgl.h"
#include <stdint.h>

void vizualize_picture(MNISTData *data, char *caption) {
  const int screenWidth = IMAGE_SIZE * SCALE_FACTOR;
  const int screenHeight = IMAGE_SIZE * SCALE_FACTOR + 200;

  InitWindow(screenWidth, screenHeight, "MNIST Image Viewer");

  SetTargetFPS(1);

  // Main game loop
  while (!WindowShouldClose()) // Detect window close button or ESC key
  {
    if (IsKeyPressed(KEY_ESCAPE))
      break;
    BeginDrawing();

    ClearBackground(RAYWHITE);

    // Draw the MNIST image
    for (int x = 0; x < IMAGE_SIZE; ++x) {
      for (int y = 0; y < IMAGE_SIZE; ++y) {
        // MNIST pixel values are 0-255, so we need to convert this to a color
        int val = data->image[y][x];
        Color color = {val, val, val, 255};
        DrawRectangle(x * SCALE_FACTOR, y * SCALE_FACTOR, SCALE_FACTOR,
                      SCALE_FACTOR, color);
      }
    }
    int textWidth = MeasureText(caption, 20);
    int xPos = (screenWidth - textWidth) / 2;
    int yPos = screenHeight - 50; // Adjust the vertical position as needed

    DrawText(caption, xPos, yPos, 20, BLACK);

    EndDrawing();
    //----------------------------------------------------------------------------------
  }

  CloseWindow(); // Close window and OpenGL context
}

void vizualize_net(NN *net) {
  const int screenWidth = 1600;
  const int screenHeight = 1000;
  const int radius = 10;

  InitWindow(screenWidth, screenHeight, "Neural net Visualization");

  SetTargetFPS(60);

  while (!WindowShouldClose()) // Detect window close button or ESC key
  {
    if (IsKeyPressed(KEY_ESCAPE))
      break;
    BeginDrawing();

    ClearBackground(RAYWHITE);

    for (int i = 0; i < net->num_layers; i++) {
      layer l = net->layers[i];
      int neurons_per_layer = l.num_neurons;
      int space_between_neurons = screenHeight / (neurons_per_layer + 1);

      for (int j = 0; j < neurons_per_layer; j++) {
        int y = (j + 1) * space_between_neurons;
        int x = (i + 1) * screenWidth / (net->num_layers + 1);
        DrawCircle(x, y, radius, DARKBLUE);

        // Draw lines (weights) to the next layer
        if (i < net->num_layers - 1) { // If not the last layer
          layer next_layer = net->layers[i + 1];
          int next_neurons_per_layer = next_layer.num_neurons;
          int next_space_between_neurons =
              screenHeight / (next_neurons_per_layer + 1);

          for (int k = 0; k < next_neurons_per_layer; k++) {
            int next_y = (k + 1) * next_space_between_neurons;
            int next_x = (i + 2) * screenWidth / (net->num_layers + 1);
            DrawLine(x, y, next_x, next_y, DARKGRAY);
          }
        }
      }
    }

    EndDrawing();
  }

  CloseWindow(); // Close window and OpenGL context
}

#define SCREEN_WIDTH 280
#define SCREEN_HEIGHT 280
#define RADIUS 10
#define IMAGE_SIZE 28

MNISTData user_draw() {
  // Initialize the window
  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Draw your number!");

  // Create a render texture (framebuffer) to hold the drawn image
  RenderTexture2D drawing = LoadRenderTexture(SCREEN_WIDTH, SCREEN_HEIGHT);

  // Clear the framebuffer to white
  BeginTextureMode(drawing);
  ClearBackground(BLACK);
  EndTextureMode();

  // Application loop
  while (!WindowShouldClose()) {
    // Check for mouse press and draw a circle of radius 10 at the mouse
    if (IsKeyPressed(KEY_ESCAPE))
      break;
    // position if pressed
    if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
      Vector2 pos = GetMousePosition();

      BeginTextureMode(drawing);
      pos.y = SCREEN_HEIGHT - pos.y;
      DrawCircleV(pos, RADIUS, WHITE);
      EndTextureMode();
    }

    // Draw the framebuffer to the screen
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawTextureEx(drawing.texture, (Vector2){0, 0}, 0.0f, 1.0f, WHITE);
    EndDrawing();
  }
  Image img = LoadImageFromTexture(drawing.texture);
  ImageResize(&img, 28, 28);
  ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);

  MNISTData m;
  for (int x = 0; x < 28; ++x) {
    for (int y = 0; y < 28; ++y) {
      m.image[x][y] = ((uint8_t *)img.data)[x * 28 + y];
    }
  }

  UnloadImage(img);
  // Cleanup
  UnloadRenderTexture(drawing);
  CloseWindow();

  return m;
}

void test_user(NN *net, MNISTData *m) {

  float ins [28*28]; 
  for(int r=0;r<28;++r){
    for(int c=0;c<28;++c){
      ins[r*28+c] = (float)m->image[r][c];
    }
  }
  // predict what it was
  predict(net, ins);
  int prediction = max_output(net);
  char *caption ;
  sprintf(caption, "The prediction is %d", prediction);

  // visualize person's drawing and say what the prediction was in the net
  vizualize_picture(m, caption);
}
